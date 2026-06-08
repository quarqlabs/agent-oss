# Argus Agent v0.2.0



import os
import json
import asyncio
import re
import numpy as np
import faiss
from datetime import datetime
from typing import TypedDict, Sequence,Optional
from dotenv import load_dotenv
import shutil
import time

from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage , ToolMessage
from langgraph.graph import StateGraph, START, END
from openai import AsyncOpenAI
from supabase import create_client, Client # 🛠️ NEW IMPORT

import tools.tool_manager as tool_manager


# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
load_dotenv()

raw_api_key = os.getenv("OPENROUTER_API_KEY")


supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
AGENT_ID = os.getenv("AGENT_ID") # 🛠️ INJECTED BY DEVOPS
USER_ID = os.getenv("USER_ID")

if not all([raw_api_key, supabase_url, supabase_key, AGENT_ID]):
    raise ValueError("Missing critical environment variables (OPENROUTER_API_KEY, SUPABASE keys, AGENT_ID).")

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
    model="google/gemini-3.1-pro-preview",
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
# HELPER: TOKEN EXTRACTION (Breakdown)
# ==========================================
def get_token_metrics(response) -> dict:
    """Extracts input, output, and total token usage from LangChain AIMessage."""
    metrics = {"input": 0, "output": 0, "total": 0}

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        metrics["input"] = response.usage_metadata.get("input_tokens", 0)
        metrics["output"] = response.usage_metadata.get("output_tokens", 0)
        metrics["total"] = response.usage_metadata.get("total_tokens", 0)
    elif hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
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
        self.memory_type = memory_type # "Semantic" or "Episodic"

    async def add_memory(self, text: str):
        if not text.strip(): return

        try:
            # 1. Get Embedding
            response = await embed_client.embeddings.create(model=EMBED_MODEL, input=text)
            if not response.data: return
            vector = response.data[0].embedding

            # 2. DEDUPLICATION CHECK: Prevent storing identical/highly similar memories
            # We run the Supabase RPC in a separate thread to prevent blocking the FastAPI event loop
            duplicate_check = await asyncio.to_thread(
                supabase.rpc("match_memories", {
                    "query_embedding": vector,
                    "match_threshold": 0.95, # 95% similarity means it's basically the same fact
                    "match_count": 1,
                    "p_agent_id": AGENT_ID,
                    "p_memory_type": self.memory_type
                }).execute
            )

            # If a near-identical memory exists, skip inserting to save DB space and tokens
            if duplicate_check.data and len(duplicate_check.data) > 0:
                print(f"🔄 [Memory] Skipped duplicate {self.memory_type} memory.")
                return

            # 3. Insert into Supabase (Non-blocking)
            await asyncio.to_thread(
                supabase.table("agent_memories").insert({
                    "agent_id": AGENT_ID,
                    "memory_type": self.memory_type,
                    "content": text,
                    "embedding": vector
                }).execute
            )
            
        except Exception as e:
            print(f"❌ [Error] Supabase embedding/insert failed: {e}")

    def clear(self):
        """Wipes this agent's memories of this type from Supabase."""
        supabase.table("agent_memories").delete().eq("agent_id", AGENT_ID).eq("memory_type", self.memory_type).execute()

    async def search(self, query: str, top_k: int = 5) -> str:
        if not query.strip(): return ""

        try:
            response = await embed_client.embeddings.create(model=EMBED_MODEL, input=query)
            if not response.data: return ""
            vector = response.data[0].embedding

            # 🛠️ Execute Supabase RPC without blocking FastAPI
            result = await asyncio.to_thread(
                supabase.rpc("match_memories", {
                    "query_embedding": vector,
                    "match_threshold": 0.30, # Ignore totally irrelevant memories
                    "match_count": top_k,
                    "p_agent_id": AGENT_ID,
                    "p_memory_type": self.memory_type
                }).execute
            )

            if not result.data:
                return ""

            results = [row["content"] for row in result.data]
            return "\n".join(results)
            
        except Exception as e:
            print(f"❌ [Error] Supabase search failed: {e}")
            return ""


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
    """Loads rules for this specific agent from Supabase."""
    try:
        res = supabase.table("agent_rules").select("*").eq("agent_id", AGENT_ID).execute()
        return res.data
    except Exception as e:
        print(f"[Warning] Failed to read procedural rules from Supabase: {e}")
        return []

def save_procedural_rules(valid_rules: list):
    """Inserts new rules into Supabase."""
    try:
        # Prepare the payloads
        payloads = []
        for rule_obj in valid_rules:
            payloads.append({
                "agent_id": AGENT_ID,
                "rule": rule_obj.get("rule"),
                "reasoning": rule_obj.get("reasoning", ""),
                "target_entity": rule_obj.get("target_entity", ""),
                "tags": rule_obj.get("tags", [])
            })
        
        if payloads:
            supabase.table("agent_rules").insert(payloads).execute()
    except Exception as e:
        print(f"[Warning] Failed to save rules to Supabase: {e}")

def wipe_all_memories():
    """Wipes all vectors and rules for this agent from the DB."""
    semantic_db.clear()
    episodic_db.clear()
    supabase.table("agent_rules").delete().eq("agent_id", AGENT_ID).execute()
    print(f"🧹 Agent {AGENT_ID} memories wiped from Supabase.")

# ==========================================
# 4. GRAPH STATE
# ==========================================
class AgentState(TypedDict):
    user_prompt: str
    chat_history: Sequence[BaseMessage]
    semantic_context: str
    episodic_context: str
    procedural_context: str
    selected_skills: list[str]      # UPDATE: Now a list of strings 
    skill_markdown: str             # NEW: Documentation for active tools
    final_response: str
    skip_learning: bool
    user_id: str            # NEW: Unique identifier for the user
    channel_type: str       # NEW: e.g., 'telegram', 'whatsapp', 'terminal'
    metrics: dict


# ==========================================
# 5. RETRIEVAL NODE (Robust Tagging)
# ==========================================
async def retrieve_memories_node(state: AgentState):
    start_time = time.time()  # START TIMER
    in_tokens = 0
    out_tokens = 0

    user_prompt = state["user_prompt"]
    history_text = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state["chat_history"][-5:]]
    )
    search_query = f"{history_text}\nUser: {user_prompt}"

    # Vector Search
    semantic_result, episodic_result = await asyncio.gather(
        semantic_db.search(search_query, top_k=5),
        episodic_db.search(search_query, top_k=5),
    )

    # Procedural Tag Routing with CoT
    all_rules = load_procedural_rules()
    procedural_result = ""

    if all_rules:

        
        known_tags = list(
            set(tag for rule in all_rules if isinstance(rule, dict) for tag in rule.get("tags", []))
        )
        tag_prompt = f"""
        You are an intelligent Routing AI. Your task is to determine which behavioral rules the agent needs to answer the user's prompt correctly.
        
        Current User Prompt: "{user_prompt}"
        Recent Context: {semantic_result}
        
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
                requested_tags = [str(tag).lower() for tag in parsed_data.get("tags", [])]
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
                    matched_rules.append(rule["rule"])

            procedural_result = "\n".join(matched_rules)

        except Exception as e:
            print(f"[Warning] Failed to parse procedural tags: {e}")
            procedural_result = ""

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
        "metrics": {
            "retrieval_in": in_tokens, 
            "retrieval_out": out_tokens
        }

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
    history_text = "\n".join([f"{m.type}: {m.content}" for m in state["chat_history"][-4:]])
    
    # --- NEW: Compile memory context for the router ---
    memory_context = f"""
    [Semantic]: {state.get('semantic_context', 'None')}
    [Episodic]: {state.get('episodic_context', 'None')}
    [Procedural]: {state.get('procedural_context', 'None')}
        """.strip()
    
    
    chosen_skills = await tool_manager.select_skills(
        user_prompt=state["user_prompt"], 
        recent_history=history_text,
        memory_context=memory_context
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
 
    print(
        f"⏱️ [Metrics] Time: {end_time - start_time:.2f}s"
    )
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

    system_instruction = f"""You are a highly advanced, disciplined AI assistant.
    
    You have three types of contextual memory available:
    [SEMANTIC - User Facts]: {state.get("semantic_context", "None")}
    [EPISODIC - Past Events]: {state.get("episodic_context", "None")}
    [PROCEDURAL - Strict Rules]: {state.get("procedural_context", "None")}

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
        system_instruction += f"\n\n[ACTIVE SKILL INSTRUCTIONS]:\n{state['skill_markdown']}"

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
                        print(f"🔧 [Loop {iteration+1}] Executing Tool: {call['name']}...")

                        # --- NEW: SECURE CONTEXT INJECTION ---
                        # Pass backend variables securely without exposing them to the LLM
                        run_config = {
                            "configurable": {
                                "user_id": state.get("user_id"),
                                "channel_type": state.get("channel_type")
                            }
                        }

                        result = fn.invoke(call["args"], config=run_config)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = "Tool not found."
                
                tool_msgs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

            
            messages.extend(tool_msgs)
            
            iteration += 1

        # --- FIX FOR PROBLEM 2: FORCE TEXT GENERATION ON TIMEOUT ---
        if iteration == MAX_ITERATIONS:
            print("⚠️ ReAct loop reached maximum iterations. Forcing final text generation.")
            # Unbind tools by using the plain `llm` so it is FORCED to output text
            messages.append(SystemMessage(content="SYSTEM INSTRUCTION: You have reached the maximum allowed tool execution limit. You must immediately provide a final text response to the user summarizing what you accomplished and what you couldn't finish. Do NOT attempt to call any more tools."))
            
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
    final_output = re.sub(r"<thinking>.*?</thinking>", "", content_str, flags=re.DOTALL).strip()
    
    # FALLBACK: If stripping thinking leaves us empty, provide the raw content or a default
    if not final_output:
        if content_str.strip():
            final_output = content_str.strip() # Show the thinking if that's all we have
        else:
            final_output = "I have processed that request using my tools, but I don't have a specific summary to display. Please let me know if you need anything else."

    end_time = time.time()
    print(f"\n⏱️ [Metrics] Generation Time: {end_time - start_time:.2f}s | Tokens: In({in_tokens}) Out({out_tokens})")

    current_metrics = state.get("metrics", {})
    current_metrics.update({
        "generation_in": in_tokens,
        "generation_out": out_tokens
    })

    # 🚀 NEW: FIRE BACKGROUND LEARNING TO REDUCE LATENCY
    if not state.get("skip_learning", False):
        asyncio.create_task(
            background_memory_update(
                state["user_prompt"],
                final_output,
                state.get("semantic_context", ""),
                state.get("episodic_context", ""),
                state.get("procedural_context", "")
            )
        )



    return {"final_response": final_output,"metrics": current_metrics}


# ==========================================
# 7. LEARNING NODE (Robust Extraction)
# ==========================================
async def learn_vector_memory(
    db: VectorMemoryManager,
    memory_type: str,
    user_prompt: str,
    ai_response: str,
    current_context: str,
) -> tuple[str, dict]:
    """Extracts facts/episodes with strict isolation and an importance threshold."""

    if memory_type == "Semantic":
        definition = "Factual, permanent information about the user (e.g., name, job, favored brands, frequently visited locations, tech stack preferences)."
        example = "User's name is Sourav, works at Argus Labs, and shops at Target."
        exclusion_rule = "DO NOT extract temporary states, greetings, or small talk."
    else:
        definition = "Significant experiences, tasks completed, purchases made, or milestone events, including the specific context of WHO, WHAT, WHERE, WHEN, and HOW."
        example = "User bought a new vacuum cleaner from Target using a 15% discount code."
        exclusion_rule = "CRITICAL: DO NOT extract greetings, pleasantries, or basic conversational filler. ONLY extract events that have long-term operational importance."

    learning_prompt = f"""
    You are a highly selective and precise Memory Extraction Agent analyzing a conversation.
    Your sole task is to extract NEW {memory_type} memory.

    DEFINITION OF {memory_type.upper()} MEMORY:
    {definition}

    Currently Retrieved {memory_type} Context: {current_context if current_context else 'None'}
    
    Interaction:
    User: {user_prompt}
    AI: {ai_response}

    STRICT EXTRACTION RULES:
    1. Extract ONLY new, significant information that matches the definition of {memory_type} memory.
    2. HIGH-FIDELITY RETENTION (CRITICAL): When extracting a memory, you MUST preserve all specific entities, locations, brand names, software tools, monetary amounts, and key parameters. Do not over-summarize. If an action happened "at Target" or "using Python", you must include those exact proper nouns in the extracted fact.
    3. {exclusion_rule}
    4. DO NOT extract information that belongs in another memory category.
    5. DO NOT prefix your sentences with labels like "Semantic:" or "Episodic:". Just state the bare fact.
    6. Example output format: {example}
    7. If the interaction lacks significant operational value or contains no new information, output exactly: NONE
    """

    response = await llm.ainvoke([HumanMessage(content=learning_prompt)])
    content = str(response.content).strip()
    m = get_token_metrics(response)

    if content and content.upper() != "NONE":
        # Extra safety check to strip accidental prefixes if the LLM misbehaves
        content = re.sub(
            r"^(Semantic|Episodic):\s*", "", content, flags=re.IGNORECASE | re.MULTILINE
        )

        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        new_lines = content.split("\n")
        saved = []
        for line in new_lines:
            if line.strip():
                formatted_line = f"{timestamp} {line.strip()}"
                await db.add_memory(formatted_line)
                saved.append(formatted_line)
        return " | ".join(saved), m
    return "", m


async def learn_procedural_memory(
    user_prompt: str, ai_response: str, current_context: str
) -> tuple[str, dict]:
    """Extracts explicit behavioral rules while strictly ignoring facts and specific task details."""

    learning_prompt = f"""
    You are a Strict Procedural Extraction Agent.
    Your ONLY job is to identify EXPLICIT, REUSABLE rules, formatting constraints, or behavioral guidelines commanded by the user.

    CRITICAL BOUNDARIES (WHAT NOT TO EXTRACT):
    1. DO NOT extract factual information about the user (e.g., their name, job title, company, or current projects). That belongs in Semantic Memory.
    2. DO NOT extract specific details of the current task (e.g., "The user needed an email about being late"). That belongs in Episodic Memory.
    3. ONLY extract a rule if the user expressed a general preference for HOW you should behave, format, or generate outputs in the future (e.g., "Always use a formal tone", "Never use emojis", "Format code in snake_case").

    ENTITY SPECIFICITY (CRITICAL RULE):
    - If the user specifies that a rule applies to a specific person (e.g., "Elias"), project, or context, you MUST explicitly include that name/context at the very beginning of the rule text (e.g., "When generating content for Elias, never use exclamation marks.").
    - If the user does not specify a target, leave it as a general global rule (e.g., "Never use exclamation marks.").

    Currently Retrieved Rules: {current_context if current_context else 'None'}
    
    Interaction:
    User: {user_prompt}
    AI: {ai_response}

    STRICT JSON OUTPUT REQUIREMENT:
    If the user explicitly established a NEW, REUSABLE procedural rule, output a JSON array of objects. 
    Each object must have:
    - "rule": The specific, universal instruction. MUST include the target entity's name if applicable.
    - "reasoning": Explain why this is a universal behavioral rule and not just a factual detail or a one-time task request.
    - "target_entity": The exact name of the person/project this applies to, or "global" if it applies to everything.
    - "tags": An array of 1-3 broad trigger words (e.g., ["python"], ["formatting"], ["Elias"]). If there is a target entity, MUST include their name as a tag to aid vector retrieval.
    
    If the user simply asked you to perform a task without giving a general behavioral rule, return exactly: NONE
    """

    response = await llm.ainvoke([HumanMessage(content=learning_prompt)])
    content = str(response.content).strip()
    m = get_token_metrics(response)

    if content and content.upper() != "NONE":
        try:
            # Clean up potential markdown formatting from the LLM
            content = content.replace("```json", "").replace("```", "").strip()
            raw_rules = json.loads(content)
            
            # ARMOR: Validate and sanitize the LLM's output
            valid_rules = []
            if isinstance(raw_rules, list):
                for item in raw_rules:
                    if isinstance(item, dict) and "rule" in item:
                        valid_rules.append(item)
            elif isinstance(raw_rules, dict) and "rule" in raw_rules:
                # LLM accidentally returned a single object instead of an array
                valid_rules.append(raw_rules)
                
            if not valid_rules:
                return "", m # Exit if no valid rules were found

            # Generate Timestamp
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

            # Prepend timestamp to each rule string
            for obj in valid_rules:
                obj["rule"] = f"{timestamp} {obj['rule']}"

            all_rules = load_procedural_rules()
            all_rules.extend(valid_rules) # Save only the validated rules
            save_procedural_rules(all_rules)

            return json.dumps(valid_rules), m
        except Exception as e:
            print(
                f"[Warning] Failed to parse new procedural rules: {e}\nContent was: {content}"
            )
    return "", m


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
    current_metrics.update({
        "learning_in": total_in,
        "learning_out": total_out
    })


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
async def background_memory_update(user_prompt: str, ai_resp: str, semantic_ctx: str, episodic_ctx: str, procedural_ctx: str):
    """Runs memory extraction silently in the background so the user doesn't wait."""
    start_time = time.time()

    results = await asyncio.gather(
        learn_vector_memory(semantic_db, "Semantic", user_prompt, ai_resp, semantic_ctx),
        learn_vector_memory(episodic_db, "Episodic", user_prompt, ai_resp, episodic_ctx),
        learn_procedural_memory(user_prompt, ai_resp, procedural_ctx),
    )

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
workflow.add_node("route_tools", route_tools_node) # NEW
workflow.add_node("generate_response", generate_response_node)
# workflow.add_node("update_memories", update_memories_node)

workflow.add_edge(START, "retrieve_memories")
workflow.add_edge("retrieve_memories", "route_tools") # NEW PATH
workflow.add_edge("route_tools", "generate_response") # NEW PATH
# workflow.add_conditional_edges("generate_response", route_after_generation)
# workflow.add_edge("update_memories", END)
workflow.add_edge("generate_response", END) # 🚀 Returns to the user instantly!

app = workflow.compile()


async def main_chat_loop():
    print("🤖 Agent V2 initialized with CoT, Vector (RAG), and Tag-Routed Memory.")
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
            "channel_type": "terminal",        # ADDED
            "metrics": {}
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
