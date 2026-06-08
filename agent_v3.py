# Argus Agent v0.3.0

import os
import json
import asyncio
import re
import numpy as np
import faiss
from datetime import datetime
from typing import TypedDict, Sequence, Optional
from dotenv import load_dotenv
import shutil
import time

from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from openai import AsyncOpenAI
from supabase import create_client, Client  # 🛠️ NEW IMPORT

import tools.tool_manager as tool_manager

from functools import wraps

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
load_dotenv()

raw_api_key = os.getenv("OPENROUTER_API_KEY")


supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
AGENT_ID = os.getenv("AGENT_ID")  # 🛠️ INJECTED BY DEVOPS
USER_ID = os.getenv("USER_ID")

if not all([raw_api_key, supabase_url, supabase_key, AGENT_ID]):
    raise ValueError(
        "Missing critical environment variables (OPENROUTER_API_KEY, SUPABASE keys, AGENT_ID)."
    )

# Initialize Supabase
supabase: Client = create_client(supabase_url, supabase_key)


# LLM for Text Generation
llm = ChatOpenAI(
    model="google/gemini-3-flash-preview",
    api_key=SecretStr(raw_api_key),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": "https://github.com/quarqlabs/argus",
        "X-Title": "Argus Agent V2",
    },
)

gen_llm = ChatOpenAI(
    model="google/gemini-3-flash-preview",
    api_key=SecretStr(raw_api_key),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": "https://github.com/quarqlabs/argus",
        "X-Title": "Argus Agent V2",
    },
)

# OpenAI Client for Vector Embeddings
embed_client = AsyncOpenAI(
    api_key=raw_api_key,
    base_url="https://openrouter.ai/api/v1",
)
EMBED_MODEL = "text-embedding-3-small"


# ==========================================
# GLOBAL CACHE & CONCURRENCY LIMITERS
# ==========================================
AGENT_CONFIG_CACHE = None
LEARNING_SEMAPHORE = asyncio.Semaphore(2)  # 🛠️ NEW: Max 2 concurrent DB saves
PENDING_LEARNING_TASKS = set()  # 🛠️ NEW: Tracks active background tasks



# ==========================================
# ROBUST NETWORK RETRY DECORATOR
# ==========================================
def network_retry(max_retries=4, initial_delay=2.0):
    """Automatically retries an async function if a network/database error occurs."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"❌ [Fatal Error] {func.__name__} failed after {max_retries} attempts: {e}")
                        raise e # Bubble up the error if it's completely dead
                    
                    print(f"⚠️ [Network Retry] {func.__name__} failed ({e}). Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(delay)
                    delay *= 2 # Exponential backoff (2s, 4s, 8s...)
        return wrapper
    return decorator


# ==========================================
# HELPER: TOKEN EXTRACTION (Breakdown)
# ==========================================
def get_token_metrics(response) -> dict:
    """Extracts input, output, and total token usage from LangChain AIMessage."""
    metrics = {"input": 0, "output": 0, "total": 0}

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        metrics["input"] = response.usage_metadata.get("input_tokens", 0)
        metrics["output"] = response.usage_metadata.get("output_tokens", 0)
        metrics["total"] = response.usage_metadata.get("total_tokens", 0)
    elif (
        hasattr(response, "response_metadata")
        and "token_usage" in response.response_metadata
    ):
        usage = response.response_metadata["token_usage"]
        # Safe check: ensures 'usage' is a dict before calling .get()
        if isinstance(usage, dict):
            metrics["input"] = usage.get("prompt_tokens", 0)
            metrics["output"] = usage.get("completion_tokens", 0)
            metrics["total"] = usage.get("total_tokens", 0)

    return metrics


# ==========================================
# 2. VECTOR DATABASE MANAGER
# ==========================================
# class VectorMemoryManager:
#     """Manages local FAISS vector DBs and Text mapping."""

#     def __init__(self, memory_type: str):
#         self.folder = f"{memory_type.lower()}_memory"
#         self.index_file = os.path.join(self.folder, "index.faiss")
#         self.text_file = os.path.join(self.folder, "memory.txt")
#         self.dim = 1536  # Dimension for text-embedding-3-small

#         os.makedirs(self.folder, exist_ok=True)
#         self.texts = self._load_texts()
#         self.index = self._load_or_create_index()

#         # ARMOR: Sync Check
#         if self.index.ntotal != len(self.texts):
#             print(f"[Warning] {memory_type} DB out of sync! FAISS has {self.index.ntotal} vectors, but memory.txt has {len(self.texts)} lines.")


#     def _load_texts(self):
#         if not os.path.exists(self.text_file):
#             return []
#         with open(self.text_file, "r", encoding="utf-8") as f:
#             return [line.strip() for line in f.readlines() if line.strip()]

#     def _load_or_create_index(self):
#         if os.path.exists(self.index_file):
#             return faiss.read_index(self.index_file)
#         return faiss.IndexFlatIP(self.dim)

#     async def add_memory(self, text: str):
#         if not text.strip():
#             return

#         try:
#             response = await embed_client.embeddings.create(
#                 model=EMBED_MODEL, input=text
#             )

#             if not response.data:
#                 print(f"[Warning] Empty embedding data received while saving memory.")
#                 return

#             vector = np.array([response.data[0].embedding], dtype="float32")
#             faiss.normalize_L2(vector)

#             self.index.add(vector)  # type: ignore
#             self.texts.append(text)

#             faiss.write_index(self.index, self.index_file)
#             with open(self.text_file, "a", encoding="utf-8") as f:
#                 f.write(text + "\n")

#         except Exception as e:
#             print(f"[Warning] Embedding API failed during learning: {e}")

#     def clear(self):
#         """Safely wipes the in-memory index and deletes files from disk."""
#         import shutil
#         self.texts = []
#         self.index = faiss.IndexFlatIP(self.dim)
#         if os.path.exists(self.folder):
#             shutil.rmtree(self.folder)
#         os.makedirs(self.folder, exist_ok=True)
#         with open(self.text_file, "w", encoding="utf-8") as f:
#             f.write("")

#     async def search(self, query: str, top_k: int = 5) -> str:
#         if self.index.ntotal == 0 or not query.strip():
#             return ""

#         try:
#             response = await embed_client.embeddings.create(
#                 model=EMBED_MODEL, input=query
#             )

#             if not response.data:
#                 print(f"[Warning] Empty embedding data received during search.")
#                 return ""

#             vector = np.array([response.data[0].embedding], dtype="float32")
#             faiss.normalize_L2(vector)

#             k = min(top_k, self.index.ntotal)
#             distances, indices = self.index.search(vector, k)  # type: ignore

#             # ARMOR: Bounds checking to prevent "list index out of range"
#             results = []
#             for idx in indices[0]:
#                 if idx != -1:
#                     if idx < len(self.texts):
#                         results.append(self.texts[idx])
#                     else:
#                         # Silently skip out-of-sync FAISS vectors instead of crashing
#                         pass

#             return "\n".join(results)

#         except Exception as e:
#             print(f"[Warning] Embedding API failed during search: {e}")
#             return ""


# New Update : In Cloud Memory
class VectorMemoryManager:
    """Manages Semantic and Episodic memories via Supabase pgvector."""

    def __init__(self, memory_type: str):
        self.memory_type = memory_type

    @network_retry(max_retries=4, initial_delay=2.0) # 🛠️ NEW: Protects DB & Embeddings
    async def execute_action(self, action: dict):
        act_type = action.get("action", "").upper()
        content = action.get("content", "")
        raw_id = action.get("id")

        # 🛠️ BULLETPROOF UUID EXTRACTION
        record_id = None
        if raw_id:
            match = re.search(r"([0-9a-fA-F\-]{36})", raw_id)
            if match:
                record_id = match.group(1)

        try:
            if act_type == "DELETE" and record_id:
                await asyncio.to_thread(
                    supabase.table("agent_memories")
                    .delete()
                    .eq("id", record_id)
                    .execute
                )
                print(f"🗑️ [Memory] DELETED {self.memory_type} memory: {record_id}")
                return

            if not content.strip():
                return

            # 🛠️ EMBED THE PURE CONTENT (No timestamps in the text!)
            response = await embed_client.embeddings.create(
                model=EMBED_MODEL, input=content
            )
            if not response.data:
                return
            vector = response.data[0].embedding

            if act_type == "ADD":
                duplicate_check = await asyncio.to_thread(
                    supabase.rpc(
                        "match_memories",
                        {
                            "query_embedding": vector,
                            "match_threshold": 0.95,
                            "match_count": 1,
                            "p_agent_id": AGENT_ID,
                            "p_memory_type": self.memory_type,
                        },
                    ).execute
                )
                if duplicate_check.data and len(duplicate_check.data) > 0:
                    print(f"🔄 [Memory] Skipped duplicate {self.memory_type} ADD.")
                    return

                await asyncio.to_thread(
                    supabase.table("agent_memories")
                    .insert(
                        {
                            "agent_id": AGENT_ID,
                            "memory_type": self.memory_type,
                            "content": content,
                            "embedding": vector,
                        }
                    )
                    .execute
                )
                print(f"✅ [Memory] ADDED {self.memory_type}: {content[:30]}...")

            elif act_type == "UPDATE" and record_id:
                await asyncio.to_thread(
                    supabase.table("agent_memories")
                    .update({"content": content, "embedding": vector})
                    .eq("id", record_id)
                    .execute
                )
                print(f"✏️ [Memory] UPDATED {self.memory_type} memory: {record_id}")

        except Exception as e:
            print(f"❌ [Error] Memory Action {act_type} failed: {e}")
            raise e # 🛠️ CRITICAL: Raise the error so the decorator catches it!

    @network_retry(max_retries=3, initial_delay=1.0) # 🛠️ NEW: Protects foreground retrieval
    async def search(self, query: str, top_k: int = 10,threshold: float = 0.50) -> str:
        if not query.strip():
            return ""
        try:
            response = await embed_client.embeddings.create(
                model=EMBED_MODEL, input=query
            )
            if not response.data:
                return ""
            vector = response.data[0].embedding

            result = await asyncio.to_thread(
                supabase.rpc(
                    "match_memories",
                    {
                        "query_embedding": vector,
                        "match_threshold": threshold,  # 🛠️ CHANGED: Strict threshold to kill noise
                        "match_count": top_k,
                        "p_agent_id": AGENT_ID,
                        "p_memory_type": self.memory_type,
                    },
                ).execute
            )

            if not result.data:
                return ""

            # 🛠️ CHANGED: Attach Database Timestamp and ID dynamically
            results = []
            for row in result.data:
                # Format DB timestamptz to string
                dt = datetime.fromisoformat(
                    row["created_at"].replace("Z", "+00:00")
                ).strftime("%Y-%m-%d %H:%M:%S")
                results.append(f"[{dt}] [ID: {row['id']}] {row['content']}")

            return "\n".join(results)

        except Exception as e:
            print(f"❌ [Error] Supabase search failed: {e}")
            raise e


    @network_retry(max_retries=3, initial_delay=2.0)
    async def clear(self):
        """Wipes all memories of this type for the current agent from Supabase."""
        try:
            await asyncio.to_thread(
                supabase.table("agent_memories")
                .delete()
                .eq("agent_id", AGENT_ID)
                .eq("memory_type", self.memory_type)
                .execute
            )
            print(f"🧹 [Memory] Wiped all {self.memory_type} memories for agent {AGENT_ID}.")
        except Exception as e:
            print(f"❌ [Error] Failed to clear {self.memory_type} memory: {e}")
            raise e # 🛠️ CRITICAL: Raise the error so the decorator catches it!

semantic_db = VectorMemoryManager("Semantic")
episodic_db = VectorMemoryManager("Episodic")


# ==========================================
# 3. PROCEDURAL MEMORY MANAGER
# ==========================================
# PROCEDURAL_FILE = "procedural_memory/rules.json"

# # Ensure it exists on initial startup
# os.makedirs("procedural_memory", exist_ok=True)
# if not os.path.exists(PROCEDURAL_FILE):
#     with open(PROCEDURAL_FILE, "w", encoding="utf-8") as f:
#         json.dump([], f)

# def load_procedural_rules():
#     """Safely loads rules, returning an empty list if the file was deleted (e.g., by benchmarks)."""
#     if not os.path.exists(PROCEDURAL_FILE):
#         return []

#     try:
#         with open(PROCEDURAL_FILE, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"[Warning] Failed to read procedural rules: {e}")
#         return []

# def save_procedural_rules(rules):
#     """Safely saves rules, ensuring the directory exists first."""
#     os.makedirs(os.path.dirname(PROCEDURAL_FILE), exist_ok=True)
#     with open(PROCEDURAL_FILE, "w", encoding="utf-8") as f:
#         json.dump(rules, f, indent=4)

# def wipe_all_memories():
#     """Wipes all vector DBs and procedural rules from RAM and Disk."""
#     import shutil
#     semantic_db.clear()
#     episodic_db.clear()

#     if os.path.exists("procedural_memory"):
#         shutil.rmtree("procedural_memory")
#     os.makedirs("procedural_memory", exist_ok=True)
#     with open(PROCEDURAL_FILE, "w", encoding="utf-8") as f:
#         json.dump([], f)
#     print("🧹 Agent RAM and Disk memories wiped clean.")


# New Update : In Cloud Memory
def load_procedural_rules() -> list:
    """Loads rules for this specific agent from Supabase, ordered by newest first."""
    try:
        # 🛠️ CHANGED: Added ordering so the newest rules are prioritized
        res = (
            supabase.table("agent_rules")
            .select("*")
            .eq("agent_id", AGENT_ID)
            .order("created_at", desc=True)
            .execute()
        )
        return res.data
    except Exception as e:
        print(f"[Warning] Failed to read procedural rules: {e}")
        return []


def get_formatted_rules_with_ids(rules: list, limit: int = 15) -> str:
    """Formats a specific list of rule objects with their IDs and DB Timestamps for LLM context."""
    if not rules:
        return ""
    limited_rules = rules[:limit]

    formatted = []
    for r in limited_rules:
        # 🛠️ Handle Timestamp formatting from Supabase for Procedural Rules
        raw_time = r.get("created_at", datetime.now().isoformat())
        dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        formatted.append(f"[{dt}] [ID: {r['id']}] {r['rule']}")

    return "\n".join(formatted)


def save_procedural_rules(valid_rules: list):
    """Inserts new rules into Supabase."""
    try:
        # Prepare the payloads
        payloads = []
        for rule_obj in valid_rules:
            payloads.append(
                {
                    "agent_id": AGENT_ID,
                    "rule": rule_obj.get("rule"),
                    "reasoning": rule_obj.get("reasoning", ""),
                    "target_entity": rule_obj.get("target_entity", ""),
                    "tags": rule_obj.get("tags", []),
                }
            )

        if payloads:
            supabase.table("agent_rules").insert(payloads).execute()
    except Exception as e:
        print(f"[Warning] Failed to save rules to Supabase: {e}")


async def wipe_all_memories():
    """Wipes all vectors and rules for this agent from the DB."""
    try:
        await semantic_db.clear()
        await episodic_db.clear()
        
        await asyncio.to_thread(
            supabase.table("agent_rules")
            .delete()
            .eq("agent_id", AGENT_ID)
            .execute
        )
        print(f"🧹 [Rules] Wiped all procedural rules for agent {AGENT_ID}.")
        print(f"✅ Agent {AGENT_ID} is now completely blank and ready for the next test.")
    except Exception as e:
        print(f"❌ [Error] Failed to wipe all memories: {e}")

def sort_memories_by_recency(memory_block: str,max_lines: int = 15) -> str:
    """Parses timestamps in a block of text and sorts lines newest-to-oldest."""
    if not memory_block or memory_block == "None":
        return "None"

    lines = memory_block.strip().split("\n")

    print(lines)

    def extract_timestamp(line):
        # Matches [YYYY-MM-DD HH:MM:SS]
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            except:
                return datetime.min
        return datetime.min

    # Sort lines by timestamp descending
    sorted_lines = sorted(lines, key=extract_timestamp, reverse=True)
    # 🛠️ TOKEN PROTECTION: Keep only the most recent N lines
    capped_lines = sorted_lines[:max_lines]
    
    return "\n".join(capped_lines)


# ==========================================
# 4. GRAPH STATE
# ==========================================
class AgentState(TypedDict):
    user_prompt: str
    chat_history: Sequence[BaseMessage]
    semantic_context: str
    episodic_context: str
    procedural_context: str
    hyde_queries: list[str]         # 🛠️ NEW: Pass queries to the next node
    selected_skills: list[str]  # UPDATE: Now a list of strings
    skill_markdown: str  # NEW: Documentation for active tools
    final_response: str
    skip_learning: bool
    user_id: str  # NEW: Unique identifier for the user
    channel_type: str  # NEW: e.g., 'telegram', 'whatsapp', 'terminal'
    metrics: dict


# ==========================================
# 5. RETRIEVAL NODE (Robust Tagging)
# ==========================================
async def retrieve_memories_node(state: AgentState):
    start_time = time.time()  # START TIMER

    # 🛠️ BENCHMARK SYNC: If this is a final benchmark question, wait for all background memories to save FIRST!
    global PENDING_LEARNING_TASKS
    if state.get("skip_learning", False) and PENDING_LEARNING_TASKS:
        print(
            f"⏳ [Benchmark Sync] Waiting for {len(PENDING_LEARNING_TASKS)} background chunks to finish saving to DB..."
        )
        await asyncio.gather(*PENDING_LEARNING_TASKS, return_exceptions=True)
        print("✅ [Benchmark Sync] All memories saved. Proceeding with retrieval.")

    in_tokens = 0
    out_tokens = 0

    user_prompt = state["user_prompt"]
    history_text = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state["chat_history"][-5:]]
    )

    # Get current date for the retrieval phase
    current_time_str = datetime.now().strftime("%A, %B %d, %Y")

    hyde_prompt = f"""
    You are an AI Search Query Optimizer. 
    Analyze the recent chat history and the user's latest prompt.
    You must generate EXACTLY FOUR distinct search queries to maximize the chances of finding the right memory in a vector database.

    Query Definitions:
    - Query 1 (Comprehensive Baseline): A concise, 3rd-person factual statement capturing the core subject of the user's intent. (This is the primary direct search).
    - Query 2 (Entity Focus / Relational Anchor):  A keyword list of specific objects, tools, brands, or places. If the prompt is relational (e.g., 'X before Y'), focus this query strictly on the anchor (Y).
    - Query 3 (Action Focus / Relational Target):  A keyword list of verbs, milestones, or thematic concepts. If the prompt is relational (e.g., 'X before Y'), focus this query strictly on the target detail (X). 
    - Query 4 (Literal Unit & Noun Net): A raw, comma-separated list of ONLY the exact nouns, numbers, and quantitative units (e.g., hours, dollars, three) from the prompt. DO NOT add the word "User". Keep it strictly to the user's literal vocabulary. DO NOT include verbs. DO NOT use synonyms. Keep it under 6 words.

    

    DO NOT answer the user's question. Just state the context for a database search.
    
    CRITICAL RULES:
    1. PERSPECTIVE: Convert 1st-person ("I", "my") into 3rd-person ("User", "User's") for Queries 1, 2, and 3. Do NOT add "User" to Query 4.
       Queries 1, 2, and 3 MUST literally start with either:
        - "User"
        - "User's"
        This is mandatory.
        Query 3 FORMAT RULES:
        - MUST start with "User's "
        - MUST contain EXACTLY 2 comma-separated concepts after "User's"
        - Each concept MUST be 1-3 words maximum
        - NO additional commas
        - NO explanations
        - NO expansions
        - Example valid outputs:
        - "User's leadership, management"
        - "User's purchases, spending"
    2. If the user asks "how many" / "total" / "count", Query 1-3 must focus on the counted action plus the object.
       Prefer verb forms like "attended", "visited", "joined", "participated" over abstract nouns like "attendance".
    3. RELATIONAL DECONSTRUCTION (CRITICAL): If the prompt relates two different things (e.g., 'What did I do before X?', 'Who was at Y with me?', 'X during Y'), Queries 2 and 3 MUST search for those two components INDEPENDENTLY. 
       - Query 2: Search for the context/anchor event (The meeting, the doctor, the concert).
       - Query 3: Search for the specific detail or action (The food, the bedtime, the companion).
    4. CONDITIONAL TIME RESOLUTION (CRITICAL): The current system date is {current_time_str}. 
       - ONLY append dates if the user's intent is temporally bound (e.g., specific past events, action times, time comparisons, durations).
       - If they use relative words (e.g., "yesterday", "last week"), calculate the EXACT absolute date/month and include it.
       - DO NOT append dates or timestamps if the user is asking about permanent facts, timeless attributes, demographics, or general preferences (e.g., "What is my ethnicity?", "Do I like dogs?").
    5. NO ANSWERS: Do not try to answer the question. Only provide the search context.
    6. JSON FORMAT ONLY: You must return a valid JSON array of 4 strings. Do not include markdown blocks, just the raw array.
    
    Recent Chat History:
    {history_text if history_text else "None"}
    
    User's Latest Prompt: "{user_prompt}"
    
    ---
    EXAMPLES OF OPTIMIZED QUERIES:

    Input: "What did I eat before the meeting?" (Relational Query)
    Output: [
        "User's food consumption prior to the meeting",
        "User's meeting history and schedule",
        "User's meals, diet, and what they ate",
        "eat, meeting"
    ]

    Input: "Where did I buy that monitor I mentioned yesterday?" (Temporal Event)
    Output: [
        "User monitor purchase, electronics stores, [Calculate Yesterday's Date]",
        "User hardware, screen, display device",
        "User shopping history, acquisitions, [Calculate Yesterday's Date]",
        "monitor, buy, yesterday"
    ]

    Input: "Where did I buy that monitor I mentioned yesterday?" (Temporal Event)
    Output: [
        "User monitor purchase, electronics stores, [Calculate Yesterday's Date]",
        "User hardware, screen, display device brand, [Calculate Yesterday's Date]",
        "User shopping history, electronics acquisition, [Calculate Yesterday's Date]",
        "monitor, buy, mentioned, yesterday"
    ]

    Input: "What is my ethnicity?" (Timeless Fact - NO DATES)
    Output: [
        "User ethnic background, heritage, ancestry",
        "User demographic identity, race, nationality",
        "User family lineage, cultural origins, genealogical roots",
        "ethnicity, ancestry"
    ]

    Input: "How many hours in total did I spend driving?" (Multi-Hop Query)
    Output: [
        "User total driving duration and travel history",
        "User vehicle, road trip, transit records",
        "User travel milestones, driving time calculation",
        "hours, road trip, destinations"
    ]

    Input: "How many projects have I led?" (Thematic)
    Output: [
        "User's history of leading and managing projects",
        "User's projects, specific assignments, deliverables",
        "User's leadership roles, management, team oversight, accomplishments",
        "projects, led, total"
    ]
    """


    hyde_response = await llm.ainvoke([HumanMessage(content=hyde_prompt)])
    content = str(hyde_response.content).strip()
    
    optimized_search_query = str(hyde_response.content).strip()

    # Track HyDE tokens
    m_hyde = get_token_metrics(hyde_response)
    in_tokens += m_hyde["input"]
    out_tokens += m_hyde["output"]

    
    # Clean JSON
    if content.startswith("```json"): content = content[7:]
    if content.startswith("```"): content = content[3:]
    if content.endswith("```"): content = content[:-3]
    content = content.strip()

    # ---------------------------------------------------------
    # 🛠️ BULLETPROOF JSON PARSING & FALLBACK LOGIC
    # ---------------------------------------------------------
    search_queries = []
    try:
        # Tier 1: Try to find the JSON array brackets, ignoring conversational filler
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            array_str = content[start_idx:end_idx+1]
            search_queries = json.loads(array_str)
            
            if not isinstance(search_queries, list):
                raise ValueError("Parsed JSON is not a list.")
        else:
            raise ValueError("No JSON array brackets found.")
            
    except Exception as e:
        print(f"⚠️ [HyDE] JSON parse failed ({e}). Attempting text fallback.")
        
        # Tier 2: Split by lines, clean numbers/bullets (e.g., "1. Query" -> "Query")
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Ignore conversational filler and empty lines
            if line and not line.lower().startswith("here are") and not line.lower().startswith("output:"):
                # Regex removes starting numbers, dots, dashes, and asterisks
                clean_line = re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
                # Strip leading/trailing quotes if the LLM added them
                clean_line = clean_line.strip('"').strip("'")
                if clean_line:
                    cleaned_lines.append(clean_line)
        
        if cleaned_lines:
            search_queries = cleaned_lines[:4]  # Take up to 4 valid lines
        else:
            # Tier 3: Ultimate Safety Net. If it's total garbage, just search the user's raw prompt.
            print("⚠️ [HyDE] Text fallback failed. Reverting to raw user prompt.")
            search_queries = [user_prompt]

    # Ensure we don't have empty queries
    search_queries = [sq for sq in search_queries if sq.strip()]
    if not search_queries:
        search_queries = [user_prompt]
    # ---------------------------------------------------------

    print("HYDE queries:")
    print(search_queries)

    # 🛠️ CHANGED: CONCURRENT SEARCH FOR ALL 4 QUERIES
    semantic_tasks = [semantic_db.search(sq, top_k=15) for sq in search_queries]
    episodic_tasks = [episodic_db.search(sq, top_k=15) for sq in search_queries]

    all_semantic_results = await asyncio.gather(*semantic_tasks)
    all_episodic_results = await asyncio.gather(*episodic_tasks)

    # 🛠️ NEW: DEDUPLICATE RESULTS BASED ON DATABASE ID
    def deduplicate_memories(results_list):
        unique_memories = {}
        for result_block in results_list:
            if not result_block: continue
            lines = result_block.split('\n')
            for line in lines:
                # Extract ID to use as a unique key
                match = re.search(r"\[ID: ([0-9a-fA-F\-]{36})\]", line)
                if match:
                    unique_memories[match.group(1)] = line
        return "\n".join(unique_memories.values())

    combined_semantic = deduplicate_memories(all_semantic_results)
    combined_episodic = deduplicate_memories(all_episodic_results)

    # Apply Temporal Sorting
    semantic_result = sort_memories_by_recency(combined_semantic,max_lines=40)
    episodic_result = sort_memories_by_recency(combined_episodic,max_lines=40)

    # Procedural Tag Routing with CoT
    all_rules = load_procedural_rules()
    procedural_result = ""

    if all_rules:

        known_tags = list(
            set(
                tag
                for rule in all_rules
                if isinstance(rule, dict)
                for tag in rule.get("tags", [])
            )
        )
        tag_prompt = f"""
        You are an intelligent Routing AI. Your task is to determine which behavioral rules the agent needs to answer the user's prompt correctly.
        
        Current User Prompt: "{user_prompt}"
        Recent Semantic Context: {semantic_result}
        Recent Episodic Context: {episodic_result}
        
        Available Rule Tags: {known_tags}
        
        CHAIN OF THOUGHT REQUIREMENTS:
        1. Analyze the user's intent. Is it a greeting? A coding request? A technical architecture question?
        2. Select tags that apply to this intent.
        3. ALWAYS include the "global" tag, as it contains universal personality traits.
        
        You MUST respond EXACTLY with a valid JSON object matching this schema:
        {{
            "reasoning": "Briefly explain what the user wants and why you selected these tags.",
            "tags": ["global", "tag1", "tag2"]
        }}
        """

        response = await llm.ainvoke([HumanMessage(content=tag_prompt)])
        # TRACK BREAKDOWN
        m = get_token_metrics(response)
        in_tokens += m["input"]
        out_tokens += m["output"]

        try:
            content = (
                str(response.content).replace("```json", "").replace("```", "").strip()
            )
            parsed_data = json.loads(content)

            # CRITICAL FIX: Verify parsed_data is a dictionary
            if isinstance(parsed_data, dict):
                requested_tags = [
                    str(tag).lower() for tag in parsed_data.get("tags", [])
                ]
            else:
                # If LLM returned a list or string, fallback to global
                requested_tags = ["global"]

            matched_rules = []
            for rule in all_rules:
                # ARMOR: Skip if the JSON file got corrupted with raw strings
                if not isinstance(rule, dict):
                    continue

                rule_tags = [t.lower() for t in rule.get("tags", [])]
                if (
                    any(tag in rule_tags for tag in requested_tags)
                    or "global" in rule_tags
                ):
                    matched_rules.append(rule)

            # 🛠️ CHANGED: Use the formatter to attach IDs and apply the Hard Limit
            procedural_result = get_formatted_rules_with_ids(matched_rules, limit=8)

        except Exception as e:
            print(f"[Warning] Failed to parse procedural tags: {e}")
            # Fallback: Just show the 10 most recent rules if routing fails
            procedural_result = get_formatted_rules_with_ids(all_rules, limit=8)

    procedural_result = sort_memories_by_recency(procedural_result)

    end_time = time.time()  # END TIMER

    print("\n--- Memory Retrieval Complete ---")
    if semantic_result:
        print(f"Semantic Found:\n{semantic_result}")
    if episodic_result:
        print(f"Episodic Found:\n{episodic_result}")
    if procedural_result:
        print(f"Procedural Found:\n{procedural_result}")
    else:
        print("Procedural Found: None (No relevant tags found)")

    print(
        f"⏱️ [Metrics] Time: {end_time - start_time:.2f}s | Tokens: In({in_tokens}) Out({out_tokens})"
    )
    print("---------------------------------\n")

    return {
        "semantic_context": semantic_result,
        "episodic_context": episodic_result,
        "procedural_context": procedural_result,
        "hyde_queries": search_queries, # 🛠️ NEW
        "metrics": {"retrieval_in": in_tokens, "retrieval_out": out_tokens},
    }


# ==========================================
# 5b. TOOL ROUTING NODE
# ==========================================
async def route_tools_node(state: AgentState):
    """Pick skills for this turn using the tool_manager."""
    start_time = time.time()

    # NEW: Disable tool calling entirely during benchmarks
    if state.get("channel_type") == "benchmark":
        print("--- Tool Routing: Skipped (Benchmark Mode) ---")
        return {"selected_skills": [], "skill_markdown": ""}

    # We pass the history to the tool router for better intent detection
    history_text = "\n".join(
        [f"{m.type}: {m.content}" for m in state["chat_history"][-4:]]
    )

    # --- NEW: Compile memory context for the router ---
    memory_context = f"""
    [Semantic]: {state.get('semantic_context', 'None')}
    [Episodic]: {state.get('episodic_context', 'None')}
    [Procedural]: {state.get('procedural_context', 'None')}
        """.strip()

    chosen_skills = await tool_manager.select_skills(
        user_prompt=state["user_prompt"],
        recent_history=history_text,
        memory_context=memory_context,
    )

    if not chosen_skills:
        print("--- Tool Routing: No skill selected ---")
        return {"selected_skills": [], "skill_markdown": ""}

    print(f"--- Tool Routing: Skills Selected -> '{chosen_skills}' ---")

    combined_markdown = ""
    for skill in chosen_skills:
        loaded = tool_manager.load_skill(skill)
        combined_markdown += f"\n### {skill.upper()} SKILL\n{loaded['markdown']}\n"

    end_time = time.time()

    print(f"⏱️ [Metrics] Time: {end_time - start_time:.2f}s")
    print("---------------------------------\n")
    return {"selected_skills": chosen_skills, "skill_markdown": combined_markdown}


# ==========================================
# 6. GENERATION NODE (Robust CoT Generation & ReAct Loop)
# ==========================================
async def generate_response_node(state: AgentState):
    start_time = time.time()

    # 1. INITIALIZE variables at the top of the function scope
    in_tokens = 0
    out_tokens = 0

    # 🛠️ TRACK CONTEXT FOR STATE UPDATES & LEARNING
    sem_ctx_to_save = state.get("semantic_context", "")
    epi_ctx_to_save = state.get("episodic_context", "")

    global AGENT_CONFIG_CACHE  # 🛠️ Pull in the global cache

    # 🛠️ 1. CACHE CHECK: Only hit Supabase if the cache is empty
    if AGENT_CONFIG_CACHE is None:
        try:
            print("🌐 [Network] Fetching Agent Config from Supabase...")
            agent_config = await asyncio.to_thread(
                supabase.table("agent_containers")
                .select(
                    "agent_name, agent_personality, agent_use_cases, agent_custom_prompt"
                )
                .eq("id", AGENT_ID)
                .single()
                .execute
            )
            AGENT_CONFIG_CACHE = agent_config.data
        except Exception as e:
            print(f"⚠️ [Warning] Failed to fetch agent config: {e}")
            AGENT_CONFIG_CACHE = (
                {}
            )  # Fallback to empty dict to prevent infinite retries

    cfg = AGENT_CONFIG_CACHE
    name = cfg.get("agent_name") or "Argus Agent"
    personality = cfg.get("agent_personality") or "professional and helpful"
    use_cases = ", ".join(cfg.get("agent_use_cases") or ["general assistance"])
    custom_prompt = cfg.get("agent_custom_prompt") or ""

    identity_instruction = f"""
    [IDENTITY & PERSONA]
    Your Name: {name}
    Personality/Tone: {personality}
    Core Objectives: You are specifically optimized for: {use_cases}.
    {f'Custom User Directives: {custom_prompt}' if custom_prompt else ''}
    

    Your responses must strictly align with this identity and tone.
                """.strip()

    system_instruction = f"""You are a highly advanced, disciplined AI assistant created by the Argus team.

    {identity_instruction}
    
    You have three types of contextual memory available:
    [SEMANTIC - User Facts]: {state.get("semantic_context", "None")}
    [EPISODIC - Past Events]: {state.get("episodic_context", "None")}
    [PROCEDURAL - Strict Rules]: {state.get("procedural_context", "None")}

    TEMPORAL TRUTH PROTOCOL (CRITICAL):
    1. The memories above are provided in RECENCY ORDER (Newest information at the top).
    2. If two memories or rules contradict each other (e.g., different names, different preferences, or opposite instructions), the NEWER memory (higher timestamp or higher in the list) is the ABSOLUTE TRUTH.
    3. Ignore obsolete or corrected information from older timestamps.

    CONFIDENCE & SYNTHESIS PROTOCOL:
    CONFIDENCE & SYNTHESIS PROTOCOL:
    1. DO NOT HEDGE. If the answer is in your memory, state it as absolute fact. Do not use phrases like "It seems" or "Based on my memory".
    2. STRICT MATH RULE: If calculating totals from multiple events, you MUST write out the step-by-step arithmetic inside your <thinking> block. 
    3. MISSING / INCOMPLETE DATA PROTOCOL (CRITICAL):
        You are strictly forbidden from assuming, guessing, estimating, or prematurely finalizing numerical counts, totals, or factual aggregations when memory retrieval may be incomplete.

        You MUST output the exact flag:
        [MISSING_DATA_FALLBACK]

        inside your <thinking> block IF ANY of the following are true:

        1. You lack sufficient evidence to fully answer the question.
        2. The question requires aggregation, counting, totals, comparisons, or synthesis across multiple memories, and you only found partial evidence.
        3. Retrieved memories contain semantically related but potentially incomplete evidence (e.g., "follows", "interested in", "volunteered at") while stronger direct evidence may still exist.
        4. You detect that the retrieved memories contain only 1-2 direct matches for a broad category question (e.g., "How many festivals", "all projects", "every trip", "total hours").
        5. You are uncertain whether retrieval coverage is exhaustive.
        6. Some retrieved memories strongly imply additional related memories may exist below the retrieval threshold.

        When using [MISSING_DATA_FALLBACK]:
        - DO NOT provide the final answer yet.
        - DO NOT estimate counts.
        - DO NOT infer totals from incomplete evidence.
        - Wait for expanded retrieval results first.


    CRITICAL EXECUTION PROTOCOL (CHAIN OF THOUGHT):
    Before answering the user, you MUST plan your response using a <thinking> block.
    
    1. Inside <thinking> ... </thinking>:
       - Analyze the user's core intent.
       - SYNTHESIS & DEDUCTION: If the user asks a specific factual question (e.g., "Where did I buy X? or when we will go Z") and the exact location/detail isn't explicitly in Episodic memory, you MUST cross-reference their Semantic habits (e.g., "User usually shops at Y or We will go to Z on x date") to deduce the highly probable answer. Connect the dots.
       - If it is a simple greeting, acknowledge it naturally without over-explaining technical facts.
       - Read the [PROCEDURAL] rules. State explicitly how you will alter your formatting or tone to obey them.
    2. MULTI-STEP TOOL USE (ReAct): If you have tools available, you can call them sequentially. If the result of a tool reveals that you need more information (e.g., an email thread references a different email), you should immediately call the tool again with the new parameters.
    3. After the </thinking> block and after ALL necessary tool calls are complete, provide your final response to the user.
    4. TOOL USE: If you use a tool, you MUST provide a helpful response to the user AFTER the tool execution is complete (e.g., "I found your email..." or "I've updated your calendar"). NEVER respond with only tool calls; always speak to the user.


    Example structure:
    <thinking>
    User is asking for a python script. I must apply the procedural rule: "No list comprehensions". I will use a standard for-loop. The tone must be dry.
    </thinking>
    Here is the Python script you requested...
    or 
    <thinking>
    User is asking where they bought a vacuum. Episodic memory says they used a 15% code. Semantic memory says they prefer buying appliances at Best Buy. I will deduce it was Best Buy. I must apply the procedural rule: "dry tone".
    </thinking>
    You purchased the vacuum from Best Buy...
    or
    <thinking>
    User wants the follow-up to the project email. I will search for the first email.
    [Tool Execution...] -> Result shows it refers to thread ID 123.
    I need to call the tool again for thread ID 123.
    [Tool Execution...] -> Result found. I will synthesize the final answer.
    </thinking>
    Here is the information from that follow-up email...
    """

    if state.get("skill_markdown"):
        system_instruction += (
            f"\n\n[ACTIVE SKILL INSTRUCTIONS]:\n{state['skill_markdown']}"
        )

    messages = (
        [SystemMessage(content=system_instruction)]
        + list(state["chat_history"])
        + [HumanMessage(content=state["user_prompt"])]
    )

    selected_skills = state.get("selected_skills", [])
    last_response = None

    if selected_skills:

        tools_list = []
        for skill in selected_skills:
            skill_data = tool_manager.load_skill(skill)
            tools_list.extend(skill_data["tools"])

        llm_with_tools = gen_llm.bind_tools(tools_list)

        # --- NEW: ReAct Loop ---
        MAX_ITERATIONS = 5  # Prevent infinite loops if the LLM gets stuck
        iteration = 0

        while iteration < MAX_ITERATIONS:

            # --- Pass 1: Intent & Initial Tool Call ---
            response = await llm_with_tools.ainvoke(messages)

            # Track Tokens
            m = get_token_metrics(response)
            in_tokens += m["input"]
            out_tokens += m["output"]
            last_response = response

            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # If the LLM didn't call any tools, it means it has formulated its final answer. Break the loop.
                break

            # CRITICAL: Must append the LLM's request to history before executing
            messages.append(response)

            tool_msgs = []
            for call in tool_calls:
                fn = next((t for t in tools_list if t.name == call["name"]), None)
                if fn:
                    try:
                        print(
                            f"🔧 [Loop {iteration+1}] Executing Tool: {call['name']}..."
                        )

                        # 🚀 NEW: CACHE INVALIDATION INTERCEPTOR
                        # If the agent uses the identity update tool, wipe the cache!
                        if call["name"] == "update_agent_identity":
                            print(
                                "🔄 [Cache] Agent identity updated. Invalidating config cache."
                            )
                            AGENT_CONFIG_CACHE = None

                        # --- NEW: SECURE CONTEXT INJECTION ---
                        # Pass backend variables securely without exposing them to the LLM
                        run_config = {
                            "configurable": {
                                "user_id": state.get("user_id"),
                                "channel_type": state.get("channel_type"),
                            }
                        }

                        result = fn.invoke(call["args"], config=run_config)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = "Tool not found."

                tool_msgs.append(
                    ToolMessage(content=str(result), tool_call_id=call["id"])
                )

            messages.extend(tool_msgs)

            iteration += 1

        # --- FIX FOR PROBLEM 2: FORCE TEXT GENERATION ON TIMEOUT ---
        if iteration == MAX_ITERATIONS:
            print(
                "⚠️ ReAct loop reached maximum iterations. Forcing final text generation."
            )
            # Unbind tools by using the plain `llm` so it is FORCED to output text
            messages.append(
                SystemMessage(
                    content="SYSTEM INSTRUCTION: You have reached the maximum allowed tool execution limit. You must immediately provide a final text response to the user summarizing what you accomplished and what you couldn't finish. Do NOT attempt to call any more tools."
                )
            )

            final_response = await gen_llm.ainvoke(messages)

            m_final = get_token_metrics(final_response)
            in_tokens += m_final["input"]
            out_tokens += m_final["output"]
            last_response = final_response

    else:
        # Normal execution
        response = await gen_llm.ainvoke(messages)
        m = get_token_metrics(response)
        in_tokens = m["input"]
        out_tokens = m["output"]
        last_response = response

    # ROBUST CONTENT PARSING: Handle strings or list-based content from OpenRouter/Gemini
    content_raw = last_response.content
    content_str = ""

    if isinstance(content_raw, list):
        for part in content_raw:
            if isinstance(part, dict):
                content_str += part.get("text", "")
            else:
                # Handle cases where the list contains raw strings
                content_str += str(part)
    else:
        content_str = str(content_raw)

    # Strip thinking block
    final_output = re.sub(
        r"<thinking>.*?</thinking>", "", content_str, flags=re.DOTALL
    ).strip()

    # =====================================================================
    # 🛠️ NEW: CORRECTIVE RAG (SELF-REFLECTIVE FALLBACK RETRIEVAL)
    # =====================================================================
    if "[MISSING_DATA_FALLBACK]" in content_str:
        print("🔄 [Self-Correction] Agent reported missing data. Triggering Fallback Retrieval with threshold 0.42...")
        
        fallback_queries = state.get("hyde_queries", [state["user_prompt"]])
        
        # Re-run search with wider net (0.42 threshold)
        fb_sem_tasks = [semantic_db.search(sq, top_k=15, threshold=0.42) for sq in fallback_queries]
        fb_epi_tasks = [episodic_db.search(sq, top_k=15, threshold=0.42) for sq in fallback_queries]
        
        fb_sem_results = await asyncio.gather(*fb_sem_tasks)
        fb_epi_results = await asyncio.gather(*fb_epi_tasks)
        
        # Deduplicate and sort using your existing helper functions
        def fallback_dedupe(results_list):
            unique = {}
            for block in results_list:
                if not block: continue
                for line in block.split('\n'):
                    match = re.search(r"\[ID: ([0-9a-fA-F\-]{36})\]", line)
                    if match: unique[match.group(1)] = line
            return "\n".join(unique.values())

        new_sem_ctx = sort_memories_by_recency(fallback_dedupe(fb_sem_results), max_lines=40)
        new_epi_ctx = sort_memories_by_recency(fallback_dedupe(fb_epi_results), max_lines=40)
        
        # 🛠️ UPDATE VARIABLES SO THEY ARE SAVED IN STATE & PASSED TO LEARNING
        sem_ctx_to_save = new_sem_ctx
        epi_ctx_to_save = new_epi_ctx

        new_instruction = system_instruction.replace(
            f"[SEMANTIC - User Facts]: {state.get('semantic_context', 'None')}", 
            f"[SEMANTIC - User Facts]: {sem_ctx_to_save}"
        ).replace(
            f"[EPISODIC - Past Events]: {state.get('episodic_context', 'None')}", 
            f"[EPISODIC - Past Events]: {epi_ctx_to_save}"
        )
        
        new_messages = [SystemMessage(content=new_instruction)] + list(state["chat_history"]) + [HumanMessage(content=state["user_prompt"])]
        
        print("🧠 [Self-Correction] New context loaded. Re-generating response...")
        fallback_response = await gen_llm.ainvoke(new_messages)
        
        # Update metrics and content
        m_fb = get_token_metrics(fallback_response)
        in_tokens += m_fb["input"]
        out_tokens += m_fb["output"]
        
        content_str = str(fallback_response.content)
        final_output = re.sub(r"<thinking>.*?</thinking>", "", content_str, flags=re.DOTALL).strip()
    # =====================================================================

    # FALLBACK: If stripping thinking leaves us empty, provide the raw content or a default
    if not final_output:
        if content_str.strip():
            final_output = (
                content_str.strip()
            )  # Show the thinking if that's all we have
        else:
            final_output = "I have processed that request using my tools, but I don't have a specific summary to display. Please let me know if you need anything else."

    end_time = time.time()
    print(
        f"\n⏱️ [Metrics] Generation Time: {end_time - start_time:.2f}s | Tokens: In({in_tokens}) Out({out_tokens})"
    )

    current_metrics = state.get("metrics", {})
    current_metrics.update({"generation_in": in_tokens, "generation_out": out_tokens})

    # 🚀 NEW: FIRE BACKGROUND LEARNING TO REDUCE LATENCY
    if not state.get("skip_learning", False):

        
        # Define a safe wrapper that respects the 4-task limit
        async def bounded_learning():
            async with LEARNING_SEMAPHORE:
                await background_memory_update(
                    state["user_prompt"],
                    final_output,
                    sem_ctx_to_save,  # 🛠️ Pass updated Semantic context here
                    epi_ctx_to_save,  # 🛠️ Pass updated Episodic context here
                    state.get("procedural_context", ""),
                )
        
        # Fire-and-forget, but track it in the global set
        task = asyncio.create_task(bounded_learning())
        PENDING_LEARNING_TASKS.add(task)
        task.add_done_callback(PENDING_LEARNING_TASKS.discard)

    # 🛠️ RETURN THE UPDATED CONTEXTS SO LANGGRAPH STATE UPDATES EVERYWHERE
    return {
            "final_response": final_output, 
            "metrics": current_metrics,
            "semantic_context": sem_ctx_to_save,
            "episodic_context": epi_ctx_to_save
        }


# ==========================================
# 7. LEARNING NODE (Robust Extraction)
# ==========================================
@network_retry(max_retries=3, initial_delay=3.0) # 🛠️ NEW: Protects LLM extraction
async def learn_vector_memory(
    db: VectorMemoryManager,
    memory_type: str,
    user_prompt: str,
    ai_response: str,
    current_context: str,
) -> tuple[str, dict]:
    """Extracts facts/episodes with strict isolation and an importance threshold."""

    if memory_type == "Semantic":
        definition = (
            "Atomic, standalone facts that build the user's core identity profile. These are permanent or long-term "
            "attributes (e.g., name, age, demographics, ethnicity, origins, job title, health/dietary needs, specific tech/creative preferences, relationships, core routines and many more such things)."
            "Extract any enduring, long-term information that defines WHO the user is, WHAT they do, or HOW they live."
        )
        instruction_extension = (
            "ATOMICITY & ENTITY-PRESERVATION RULE (CRITICAL): Separate unrelated facts, BUT NEVER sever relational links. "
            "HEURISTIC: If the information reveals the user's background, origins, physical/mental traits, or strict lifestyle parameters (things that will likely still be true 5 years from now), you MUST extract it as a standalone fact. "
            "Additionally, extract CURRENT ACTIVE STATUSES or TRACKED INVENTORIES. "
            "Even if a status is not permanent, if it involves a specific count, a pending logistical requirement, or a multi-stage endeavor, it must be captured as a fact. "
            "Example: If the user mentions a specific number of items in a certain stage of a process, or a count of assets they are managing, record that specific quantity and status."
            "If the user says 'I am 22 and my friend Sarah likes destiny', issue TWO 'ADD' actions: 1. 'User is 22 years old.' 2. 'User has a friend named Sarah who is interested in destiny.' "
            "NEVER replace specific names with generic pronouns like 'a friend' or 'they'."
        )
        example = (
            '{ "action": "ADD", "content": "User\'s name is Sourav." }, '
            '{ "action": "ADD", "content": "User works at Argus Labs." }'
            '{ "action": "ADD", "content": "User\'s high school friend is named Sarah." }, '
        )
        exclusion_rule = "DO NOT extract temporary states, emotions, or the narrative of the conversation. DO NOT extract greetings, pleasantries, or basic conversational filler."
    else:
        definition = (
            "The narrative occurrence of the conversation. Focus on WHAT happened during the "
            "interaction (e.g., User introduced themselves, Agent provided code, User corrected a mistake)."
        )
        instruction_extension = (
            "EVENT-ONLY & ENTITY-PRESERVATION RULE (CRITICAL): Focus on the milestone achieved in the conversation. "
            "HOWEVER, you MUST preserve specific names, identities, and entities within the event. "
            "If the user had a conversation with 'Sarah' about 'destiny', you MUST write 'User had a conversation with their friend Sarah about destiny.' DO NOT generalize to 'a friend'. Do not duplicate facts that belong in Semantic memory."
        )
        example = (
            '{ "action": "ADD", "content": "User introduced themselves for the first time and established their professional role." }'
            '{ "action": "ADD", "content": "User transitioned from exploring existential theory to actively applying meaning after a conversation with their friend Sourav." }'
            '{ "action": "ADD", "content": "User is doing xyz project about xxx" }'
        )
        exclusion_rule = "CRITICAL: DO NOT extract temporary states, greetings, or small talk. (Note: You ARE allowed and encouraged to include specific names, brands, or places if they are directly involved in the event)."

    # 🛠️ NEW: Pass current time to resolve relative dates
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    learning_prompt = f"""
    You are a Cognitive Memory Editor managing a {memory_type} database.
    Your job is to consolidate information by issuing ADD, UPDATE, or DELETE commands.

    DEFINITION OF {memory_type.upper()} MEMORY:
    {definition}

    {instruction_extension}

    CURRENT ACTIVE MEMORIES (With Database IDs):
    {current_context if current_context else 'None'}
    
    Interaction:
    User: {user_prompt}
    AI: {ai_response}


    CRITICAL TIME RESOLUTION: The current system time is {current_time}. 
    If the user uses relative time words (e.g., "yesterday", "last month", "tomorrow"), you MUST convert them into absolute dates within the "content" string you generate. 
    Example: If user says "I started a diet yesterday", store "User started a diet on [Calculated Date]."


    STRICT EXTRACTION & CONSOLIDATION RULES:
    1. ATOMICITY: Each 'ADD' or 'UPDATE' action must contain exactly ONE independent piece of information.
    2. TYPE ISOLATION: {exclusion_rule}
    3. HIGH-FIDELITY: Preserve specific entities, brand names, and proper nouns (for Semantic). 
    4. NEW INFO: If new information is provided that doesn't exist, use "ADD".
    5. CORRECTIONS: If the user corrects or changes past information, you MUST use "UPDATE" with the ID of the old memory to overwrite it. Do NOT use ADD for corrections.
    6. REVOCATION: If the user explicitly revokes information, use "DELETE" on the old ID.
    7. NO PREFIXES: Do not use "Semantic:" or "Episodic:" labels in the content.
    8. NO GREETINGS: Ignore "Hello", "How are you", etc.

    OUTPUT FORMAT:
    You must return a raw JSON object with an "actions" array. Return exactly `{{"actions": []}}` if no changes are needed.
    {{
        "actions": [
            {{ "action": "ADD", "content": "Fact or Event 1" }},
            {{ "action": "ADD", "content": "Fact or Event 2" }},
            {{ "action": "UPDATE", "id": "uuid", "content": "Updated Fact" }},
            {{ "action": "DELETE", "id": "uuid" }}
        ]
    }}
    Example for this task:
    {example}
    """
    # print("learning_prompt:",learning_prompt)

    response = await llm.ainvoke([HumanMessage(content=learning_prompt)])
    # print("response :",response)
    content = str(response.content).strip()

    # print("data:",content, current_context)
    m = get_token_metrics(response)

    # Clean up common LLM markdown wrappers
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    actions_executed = 0

    if content and content.upper() != "NONE" and content != '{"actions": []}':
        try:
            # 1. Attempt strict JSON parsing
            data = json.loads(content)
            actions = data.get("actions", [])

            for act in actions:
                # 🛠️ REMOVED THE TIMESTAMP INJECTION LOGIC HERE
                # if "content" in act and not act["content"].startswith("["):
                #     timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                #     act["content"] = f"{timestamp} {act['content']}"

                await db.execute_action(act)
                actions_executed += 1

        except json.JSONDecodeError:
            # 2. FALLBACK PARSER: If the LLM just dumped raw text instead of JSON
            print(
                f"⚠️ [Warning] {memory_type} Editor returned raw text, falling back to ADD action."
            )

            # Strip accidental prefixes
            clean_content = re.sub(
                r"^(Semantic|Episodic):\s*",
                "",
                content,
                flags=re.IGNORECASE | re.MULTILINE,
            )

            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            lines = clean_content.split("\n")

            for line in lines:
                if line.strip():
                    fallback_act = {
                        "action": "ADD",
                        "content": line.strip(),  # 🛠️ REMOVED THE TIMESTAMP INJECTION LOGIC HERE
                    }
                    await db.execute_action(fallback_act)
                    actions_executed += 1

        except Exception as e:
            print(f"❌ [Error] Memory Execution failed: {e}")
            raise e  # 🛠️ ADD THIS: Signal the decorator to retry the whole process

    return actions_executed, m

@network_retry(max_retries=3, initial_delay=3.0) # 🛠️ NEW: Protects LLM extraction
async def learn_procedural_memory(
    user_prompt: str, ai_response: str, current_context: str
) -> tuple[int, dict]:
    """Extracts, updates, or deletes explicit behavioral rules with strict formatting constraints."""

    learning_prompt = f"""
    You are a Cognitive Procedural Database Editor.
    Your job is to identify, consolidate, and clean up behavioral rules, personas, and formatting constraints.

    CRITICAL BOUNDARIES (WHAT NOT TO EXTRACT):
    1. DO NOT extract factual information about the user (e.g., their name, job, or projects). That belongs in Semantic Memory.
    2. DO NOT extract specific details of the current task (e.g., "The user needed an email about being late"). That belongs in Episodic Memory.
    3. ONLY extract a rule if the user expressed a general preference for HOW you should behave, format, or generate outputs in the future (e.g., "Always use a formal tone", "Never use emojis", "Format code in snake_case").

    ENTITY SPECIFICITY (CRITICAL RULE):
    - If the user specifies that a rule applies to a specific person (e.g., "Elias"), project, or context, you MUST explicitly include that name/context at the very beginning of the rule text (e.g., "When generating content for Elias..."). Set the "target_entity" field to this name.
    - If no target is specified, leave "target_entity" as "global".

    CURRENT ACTIVE RULES (With Database IDs):
    {current_context if current_context else 'None'}
    
    Interaction:
    User: {user_prompt}
    AI: {ai_response}

    STRICT CONSOLIDATION RULES:
    1. If the user establishes a NEW rule, use "ADD".
    2. CONTRADICTION REMOVAL & LOSSY-UPDATE PREVENTION (CRITICAL): If the user corrects a past rule (e.g., "Stop calling me that" or fixing a typo like 'todasy' to 'today'), you MUST use "UPDATE" and provide the ID of the old rule. When updating, you MUST preserve all existing specific entities and examples from the old rule.  DO NOT "ADD" the correction, as leaving the old rule active will confuse the generation agent.
         - Example: If an old rule says "Avoid allergens (e.g., peanuts, dairy)" and the user adds "gluten", the updated rule MUST say "(e.g., peanuts, dairy, gluten)". Do not drop old examples unless explicitly revoked.
    3. If the user explicitly revokes a rule, use "DELETE" on the old ID.

    OUTPUT FORMAT:
    Return a raw JSON object with an "actions" array. Return exactly `{{"actions": []}}` if no changes are needed.
    {{
        "actions": [
            {{
                "action": "ADD", 
                "rule": "[Timestamp] New rule here.",
                "target_entity": "global",
                "tags": ["formatting"],
                "reasoning": "Why this rule was added."
            }},
            {{
                "action": "UPDATE", 
                "id": "uuid-from-context",
                "rule": "[Timestamp] Corrected rule here.",
                "target_entity": "global",
                "tags": ["formatting"],
                "reasoning": "Why it was updated."
            }},
            {{
                "action": "DELETE", 
                "id": "uuid-to-delete",
                "reasoning": "Why it was removed."
            }}
        ]
    }}
    """

    response = await llm.ainvoke([HumanMessage(content=learning_prompt)])
    content = str(response.content).strip()
    m = get_token_metrics(response)

    # Clean up common LLM markdown wrappers
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    actions_executed = 0

    if content and content.upper() != "NONE" and content != '{"actions": []}':
        try:
            # 1. Attempt strict JSON parsing
            data = json.loads(content)
            actions = data.get("actions", [])

            for act in actions:
                act_type = act.get("action", "").upper()
                raw_id = act.get("id")

                # 🛠️ BULLETPROOF UUID EXTRACTION
                record_id = None
                if raw_id:
                    match = re.search(r"([0-9a-fA-F\-]{36})", raw_id)
                    if match:
                        record_id = match.group(1)

                # 🛠️ REMOVE THIS BLOCK ENTIRELY
                # # Inject Timestamp securely if missing
                # if "rule" in act and not act["rule"].startswith("["):
                #     timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                #     act["rule"] = f"{timestamp} {act['rule']}"

                if act_type == "DELETE" and record_id:
                    await asyncio.to_thread(
                        supabase.table("agent_rules")
                        .delete()
                        .eq("id", record_id)
                        .execute
                    )
                    print(f"🗑️ [Rules] DELETED rule: {record_id}")
                    actions_executed += 1

                elif act_type == "ADD" and "rule" in act:
                    await asyncio.to_thread(
                        supabase.table("agent_rules")
                        .insert(
                            {
                                "agent_id": AGENT_ID,
                                "rule": act.get("rule"),
                                "reasoning": act.get("reasoning", ""),
                                "target_entity": act.get("target_entity", "global"),
                                "tags": act.get("tags", ["general"]),
                            }
                        )
                        .execute
                    )
                    print(f"✅ [Rules] ADDED new rule.")
                    actions_executed += 1

                elif act_type == "UPDATE" and record_id and "rule" in act:
                    await asyncio.to_thread(
                        supabase.table("agent_rules")
                        .update(
                            {
                                "rule": act.get("rule"),
                                "reasoning": act.get("reasoning", ""),
                                "target_entity": act.get("target_entity", "global"),
                                "tags": act.get("tags", ["general"]),
                            }
                        )
                        .eq("id", record_id)
                        .execute
                    )
                    print(f"✏️ [Rules] UPDATED rule: {record_id}")
                    actions_executed += 1

        except json.JSONDecodeError:
            # 2. FALLBACK PARSER: If the LLM dumped raw text instead of JSON
            print(
                f"⚠️ [Warning] Procedural Editor returned raw text, falling back to basic ADD action."
            )

            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            lines = content.split("\n")

            for line in lines:
                if line.strip() and len(line.strip()) > 10:
                    try:
                        await asyncio.to_thread(
                            supabase.table("agent_rules")
                            .insert(
                                {
                                    "agent_id": AGENT_ID,
                                    "rule": f"{timestamp} {line.strip()}",
                                    "reasoning": "Fallback extraction",
                                    "target_entity": "global",
                                    "tags": ["fallback"],
                                }
                            )
                            .execute
                        )
                        actions_executed += 1
                    except Exception as fallback_err:
                        print(
                            f"❌ [Error] Fallback Rule Insertion failed: {fallback_err}"
                        )

        except Exception as e:
            print(f"❌ [Error] Rule Execution failed: {e}")
            raise e  # 🛠️ ADD THIS: Signal the decorator to retry the whole process

    return actions_executed, m


async def update_memories_node(state: AgentState):
    start_time = time.time()

    u_prompt = state["user_prompt"]
    ai_resp = state["final_response"]

    results = await asyncio.gather(
        learn_vector_memory(
            semantic_db,
            "Semantic",
            u_prompt,
            ai_resp,
            state.get("semantic_context", ""),
        ),
        learn_vector_memory(
            episodic_db,
            "Episodic",
            u_prompt,
            ai_resp,
            state.get("episodic_context", ""),
        ),
        learn_procedural_memory(u_prompt, ai_resp, state.get("procedural_context", "")),
    )

    # Unpack the tuples
    sem_content, sem_tokens = results[0]
    epi_content, epi_tokens = results[1]
    pro_content, pro_tokens = results[2]

    # Unpack and Sum
    total_in = sum(r[1]["input"] for r in results)
    total_out = sum(r[1]["output"] for r in results)

    # Fetch existing metrics (from retrieval + generation) and add learning stats
    current_metrics = state.get("metrics", {})
    current_metrics.update({"learning_in": total_in, "learning_out": total_out})

    end_time = time.time()  # END TIMER

    print("\n--- Memory Learning Complete ---")
    if sem_content:
        print(f"💡 Learned Semantic: {sem_content}")
    if epi_content:
        print(f"💡 Learned Episodic: {epi_content}")
    if pro_content:
        print(f"💡 Learned Procedural: {pro_content}")
    if not any([sem_content, epi_content, pro_content]):
        print("No new memories learned this turn.")

    print(
        f"⏱️ [Metrics] Learning Time: {end_time - start_time:.2f}s | Total Tokens: In({total_in}) Out({total_out})"
    )
    print("--------------------------------\n")
    return {"metrics": current_metrics}


# ==========================================
# 7b. BACKGROUND LEARNING PROCESS
# ==========================================
async def background_memory_update(
    user_prompt: str,
    ai_resp: str,
    semantic_ctx: str,
    episodic_ctx: str,
    procedural_ctx: str,
):
    """Runs memory extraction silently in the background so the user doesn't wait."""
    start_time = time.time()

    results = await asyncio.gather(
        learn_vector_memory(
            semantic_db, "Semantic", user_prompt, ai_resp, semantic_ctx
        ),
        learn_vector_memory(
            episodic_db, "Episodic", user_prompt, ai_resp, episodic_ctx
        ),
        learn_procedural_memory(user_prompt, ai_resp, procedural_ctx),
    )

    # print("result:" ,results)

    # Unpack the tuples
    sem_content, sem_tokens = results[0]
    epi_content, epi_tokens = results[1]
    pro_content, pro_tokens = results[2]

    total_in = sum(r[1]["input"] for r in results)
    total_out = sum(r[1]["output"] for r in results)

    end_time = time.time()

    print("\n--- Background Memory Learning Complete ---")
    if sem_content:
        print(f"💡 Learned Semantic: {sem_content}")
    if epi_content:
        print(f"💡 Learned Episodic: {epi_content}")
    if pro_content:
        print(f"💡 Learned Procedural: {pro_content}")
    if not any([sem_content, epi_content, pro_content]):
        print("No new memories learned this turn.")

    print(
        f"⏱️ [Metrics] Background Learning Time: {end_time - start_time:.2f}s | Total Tokens: In({total_in}) Out({total_out})"
    )
    print("--------------------------------\n")


# ==========================================
# 8. BUILD GRAPH & INTERFACES
# ==========================================
def route_after_generation(state: AgentState):
    return END if state.get("skip_learning", False) else "update_memories"


workflow = StateGraph(AgentState)
workflow.add_node("retrieve_memories", retrieve_memories_node)
workflow.add_node("route_tools", route_tools_node)  # NEW
workflow.add_node("generate_response", generate_response_node)
# workflow.add_node("update_memories", update_memories_node)

workflow.add_edge(START, "retrieve_memories")
workflow.add_edge("retrieve_memories", "route_tools")  # NEW PATH
workflow.add_edge("route_tools", "generate_response")  # NEW PATH
# workflow.add_conditional_edges("generate_response", route_after_generation)
# workflow.add_edge("update_memories", END)
workflow.add_edge("generate_response", END)  # 🚀 Returns to the user instantly!

app = workflow.compile()


async def main_chat_loop():
    print(
        "🤖 Argus Agent  V3 - Cognitive Memory Editor, Temporal Truth Protocol, and Background Learning"
    )
    chat_history = []
    while True:
        u_input = input("\nYou: ")
        if u_input.lower() in ["exit", "quit"]:
            break

        initial_state: AgentState = {
            "user_prompt": u_input,
            "chat_history": chat_history,
            "semantic_context": "",
            "episodic_context": "",
            "procedural_context": "",
            "selected_skills": [],
            "skill_markdown": "",
            "final_response": "",
            "skip_learning": False,
            "user_id": "local_terminal_user",  # ADDED
            "channel_type": "terminal",  # ADDED
            "metrics": {},
        }
        final_state = await app.ainvoke(initial_state)
        print(f"\n🤖 Agent: {final_state['final_response']}")

        chat_history.extend(
            [
                HumanMessage(content=u_input),
                AIMessage(content=final_state["final_response"]),
            ]
        )


if __name__ == "__main__":
    asyncio.run(main_chat_loop())
