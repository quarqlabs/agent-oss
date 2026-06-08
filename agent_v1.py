# Argus Agent v0.1.0


import os
import asyncio
from datetime import datetime
from typing import TypedDict, Sequence
from urllib import response
from dotenv import load_dotenv


import time


from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. NEW IMPORT: Needed to satisfy the SecretStr type requirement
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Load environment variables from the .env file
load_dotenv()

# Verify the key was loaded (optional, helps with debugging)
# if not os.environ.get("GOOGLE_API_KEY"):
#     raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")


# 2. Extract and Validate the Key
raw_api_key = os.getenv("OPENROUTER_API_KEY")
if not raw_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment.")


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
        metrics["input"] = usage.get("prompt_tokens", 0)
        metrics["output"] = usage.get("completion_tokens", 0)
        metrics["total"] = usage.get("total_tokens", 0)

    return metrics


# ==========================================
# 2. DEFINE GRAPH STATE
# ==========================================
class AgentState(TypedDict):
    user_prompt: str
    chat_history: Sequence[BaseMessage]
    semantic_context: str
    episodic_context: str
    procedural_context: str
    final_response: str
    skip_learning: bool


# Initialize the Gemini Model
# llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

# 3. UPDATED: Initialize the LLM via OpenRouter
# Note: Use the OpenRouter model ID (e.g., "google/gemini-2.0-flash-001")
llm = ChatOpenAI(
    model="google/gemini-3-flash-preview",
    api_key=SecretStr(raw_api_key),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": "https://github.com/quarqlabs/argus",  # Optional: Your site URL for OpenRouter rankings
        "X-Title": "Argus Agent",  # Optional: Your App name for OpenRouter rankings
    },
)


# ==========================================
# 3. MEMORY RETRIEVAL (ASYNC WORKERS)
# ==========================================
async def extract_relevant_memory(
    memory_type: str, folder_path: str, prompt: str, history_text: str
) -> tuple[str, dict]:
    """Reads all files in a memory folder and uses AI to extract relevant rows."""

    # 1. Read all lines from all index files in the folder
    all_lines = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith("01.txt"):
                with open(
                    os.path.join(folder_path, filename), "r", encoding="utf-8"
                ) as f:
                    all_lines.extend(f.readlines())

    if not all_lines:
        return "", {"input": 0, "output": 0, "total": 0}

    memory_text = "".join(all_lines)

    # 2. Prompt the AI to filter the lines
    filter_prompt = f"""
    You are a memory retrieval assistant. Your job is to look at a list of memories and extract ONLY the exact lines that are relevant to the user's current prompt and conversation history.
    If no lines are relevant, return an empty string. DO NOT invent memories. DO NOT converse. Return exact rows.

    CONTEXTUAL FILTERING RULES:
    1. If the user is GREETING you (e.g., "hello", "hi", "how are you"), ONLY retrieve their Name and Title. IGNORE technical rules, past outages, or coding preferences.
   

    Memory Category: {memory_type}
    Recent History: {history_text}
    Current User Prompt: "{prompt}"
    
    Database Lines:
    {memory_text}

    Task: Return only the relevant lines. If the prompt is a simple greeting or something that is not related to the current intent, be minimal and to the point. Return "NONE" if nothing is strictly relevant to the current intent.
    """

    # Run the LLM call asynchronously
    response = await llm.ainvoke([HumanMessage(content=filter_prompt)])
    metrics = get_token_metrics(response)
    # Handle both string and list responses
    content = response.content
    # SAFELY PARSE CONTENT: Handle if LangChain returns a list/dict of blocks instead of a string
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and "text" in part
        )
    elif isinstance(content, dict):
        content = content.get("text", "")

    return str(content).strip(), metrics


async def retrieve_memories_node(state: AgentState):
    """LangGraph Node: Fires off 3 async threads to get context from memory files."""

    start_time = time.time()
    user_prompt = state["user_prompt"]

    # Convert history to a readable string for the filter prompt
    history_text = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state["chat_history"][-4:]]
    )  # Send last 4 turns for context

    # Run all three memory extractions CONCURRENTLY (Async Threads)
    results = await asyncio.gather(
        extract_relevant_memory(
            "Semantic (Factual info about user)",
            "semantic_memory",
            user_prompt,
            history_text,
        ),
        extract_relevant_memory(
            "Episodic (Past experiences/actions of user and ai agent)",
            "episodic_memory",
            user_prompt,
            history_text,
        ),
        extract_relevant_memory(
            "Procedural (Rules/Preferences/Instructions/guidlines for how the agent should behave)",
            "procedural_memory",
            user_prompt,
            history_text,
        ),
    )

    # Sum metrics
    total_in = sum(r[1]["input"] for r in results)
    total_out = sum(r[1]["output"] for r in results)

    end_time = time.time()

    print("\n--- Memory Retrieval Complete ---")
    if results[0]:
        print(f"Semantic Found: {results[0]}")
    if results[1]:
        print(f"Episodic Found: {results[1]}")
    if results[2]:
        print(f"Procedural Found: {results[2]}")
    print(
        f"⏱️ [Metrics] Retrieval Time: {end_time - start_time:.2f}s | Tokens: In({total_in}) Out({total_out})"
    )
    print("---------------------------------\n")

    return {
        "semantic_context": results[0],
        "episodic_context": results[1],
        "procedural_context": results[2],
    }


# ==========================================
# 4. GENERATE FINAL RESPONSE
# ==========================================
async def generate_response_node(state: AgentState):
    """LangGraph Node: Assembles the final prompt with memories and gets the response."""

    start_time = time.time()

    # Construct the System Prompt with retrieved memories
    system_instruction = f"""You are a highly disciplined AI assistant.
    
    CRITICAL OPERATING CONSTRAINTS:
    Below are three types of memories. You MUST prioritize [PROCEDURAL MEMORY]. 
    These are not suggestions; they are strict formatting and behavioral requirements.
    
    1. Analyze the [PROCEDURAL MEMORY] first. 
    2. Identify any "Before you do X, do Y" or "Always do Z" rules.
    3. Plan your response structure to satisfy all rules simultaneously.
    4. Execute the response.


    INSTRUCTION ON HOW TO USE MEMORY:
    1. DO NOT be robotic. Be a helpful colleague.
    2. MATCH THE INTENT: 
       - If the prompt is casual (Hi/Hello), respond normally, Use his/her name if known. 
       - Do NOT use technical headers or "Post-Mortem" formatting for simple greetings.
    3. PROCEDURAL ADHERENCE: Use procedural rules ONLY when the context demands it. 
       - (e.g., Use the SQL rule ONLY if writing SQL. Use the Safety rule ONLY if providing shell commands).

    Use the following contextual memories to inform your response. If a memory is empty, ignore it.
    
    [SEMANTIC MEMORY - Facts about the user]:
    {state.get("semantic_context", "None")}
    
    [EPISODIC MEMORY - Past experiences/conversations]:
    {state.get("episodic_context", "None")}
    
    [PROCEDURAL MEMORY - STRICT RULES & Instructions for how you should behave]:
    {state.get("procedural_context", "None")}

    
    """

    # Build the final message list
    messages = (
        [SystemMessage(content=system_instruction)]
        + list(state["chat_history"])
        + [HumanMessage(content=state["user_prompt"])]
    )

    # Generate the final response
    response = await llm.ainvoke(messages)
    metrics = get_token_metrics(response)

    end_time = time.time()
    print(
        f"⏱️ [Metrics] Generation Time: {end_time - start_time:.2f}s | Tokens: In({metrics['input']}) Out({metrics['output']})"
    )

    content = response.content

    # SAFELY PARSE CONTENT: Handle if LangChain returns a list/dict of blocks instead of a string
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and "text" in part
        )
    elif isinstance(content, dict):
        content = content.get("text", "")

    return {"final_response": str(content).strip()}


# ==========================================
# NEW: MEMORY LEARNING & UPDATING
# ==========================================
async def learn_new_memory(
    memory_type: str, folder_path: str, user_prompt: str, ai_response: str
) -> tuple[str, dict]:
    """Evaluates the recent interaction and extracts new memories to save."""

    # 1. Read existing memories to provide context (so it doesn't save duplicates)
    all_lines = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(
                    os.path.join(folder_path, filename), "r", encoding="utf-8"
                ) as f:
                    all_lines.extend(f.readlines())

    existing_memories = "".join(all_lines)

    # 2. Prompt the AI to identify ONLY new information
    learning_prompt = f"""
    You are an AI memory extraction agent. Your task is to extract NEW information from the latest conversation turn to save into {memory_type} memory.

    Memory Category Definitions:
    - Semantic: Factual, permanent information about the user (e.g., name, job, location).
    - Episodic: Experiences, actions taken, or a summary of what just happened in this specific interaction.
    - Procedural: Rules, formatting preferences, or instructions on how the agent should behave.

    Currently stored {memory_type} memories:
    {existing_memories if existing_memories else 'No memories yet.'}

    Latest Interaction:
    User Prompt: {user_prompt}
    AI Response: {ai_response}

    Instructions:
    1. Analyze the Latest Interaction. Is there any NEW information that belongs in {memory_type} memory that is NOT already in the stored memories?
    2. If there is new information, write each distinct new piece of information as a single, concise sentence on a new line.
    3. DO NOT include timestamps (I will add them). DO NOT include conversational filler.
    4. If there is NO new information to add for this category, output exactly the word: NONE
    """

    response = await llm.ainvoke([HumanMessage(content=learning_prompt)])
    metrics = get_token_metrics(response)
    content = response.content

    # Safely parse content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and "text" in part
        )
    elif isinstance(content, dict):
        content = content.get("text", "")

    content = str(content).strip()

    # 3. If new memories were found, append them to the file with a timestamp
    if content and content.upper() != "NONE":
        new_lines = content.split("\n")
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        file_path = os.path.join(folder_path, "index_01.txt")
        os.makedirs(folder_path, exist_ok=True)

        saved_lines = []
        with open(file_path, "a", encoding="utf-8") as f:
            for line in new_lines:
                line = line.strip()
                if line:
                    f.write(f"{timestamp} {line}\n")
                    saved_lines.append(line)

        return " | ".join(saved_lines), metrics

    return "", metrics


async def update_memories_node(state: AgentState):
    """LangGraph Node: Fires off 3 async threads to learn and save new memories."""

    start_time = time.time()
    user_prompt = state["user_prompt"]
    ai_response = state["final_response"]

    # Run all three memory learning extractions CONCURRENTLY
    results = await asyncio.gather(
        learn_new_memory("Semantic", "semantic_memory", user_prompt, ai_response),
        learn_new_memory("Episodic", "episodic_memory", user_prompt, ai_response),
        learn_new_memory("Procedural", "procedural_memory", user_prompt, ai_response),
    )

    total_in = sum(r[1]["input"] for r in results)
    total_out = sum(r[1]["output"] for r in results)

    end_time = time.time()

    print("\n--- Memory Learning Complete ---")
    if results[0]:
        print(f"💡 Learned Semantic: {results[0]}")
    if results[1]:
        print(f"💡 Learned Episodic: {results[1]}")
    if results[2]:
        print(f"💡 Learned Procedural: {results[2]}")
    if not any(results):
        print("No new memories learned this turn.")

    print(
        f"⏱️ [Metrics] Learning Time: {end_time - start_time:.2f}s | Total Tokens: In({total_in}) Out({total_out})"
    )
    print("--------------------------------\n")

    return {}  # State doesn't need to change, we just wrote to files


# ==========================================
# 5. BUILD THE LANGGRAPH
# ==========================================


def route_after_generation(state: AgentState):
    """Router: Decides whether to learn or end the turn immediately."""
    if state.get("skip_learning", False):
        return END
    return "update_memories"


workflow = StateGraph(AgentState)

workflow.add_node("retrieve_memories", retrieve_memories_node)
workflow.add_node("generate_response", generate_response_node)
workflow.add_node("update_memories", update_memories_node)  # NEW NODE

workflow.add_edge(START, "retrieve_memories")
workflow.add_edge("retrieve_memories", "generate_response")

# NEW CONDITIONAL EDGE: Only go to update_memories if skip_learning is False
workflow.add_conditional_edges("generate_response", route_after_generation)
workflow.add_edge("update_memories", END)

app = workflow.compile()


# Per-user chat history storage (keyed by telegram_id)
_user_chat_histories: dict[str, list[BaseMessage]] = {}


async def get_agent_response(user_prompt: str, telegram_id: str) -> str:
    """
    Public API for the Argus Agent. Call this from FastAPI/Telegram webhook.
    Manages per-user chat history automatically.

    Args:
        user_prompt: The user's message text.
        telegram_id: Unique Telegram user ID (used to isolate chat history).

    Returns:
        The agent's response string.
    """
    # Get or create chat history for this user
    if telegram_id not in _user_chat_histories:
        _user_chat_histories[telegram_id] = []

    chat_history = _user_chat_histories[telegram_id]

    # Prepare the initial state
    initial_state: AgentState = {
        "user_prompt": user_prompt,
        "chat_history": chat_history,
        "semantic_context": "",
        "episodic_context": "",
        "procedural_context": "",
        "final_response": "",
        "skip_learning": True,
    }

    # Run the full LangGraph pipeline
    final_state = await app.ainvoke(initial_state)

    agent_response = final_state["final_response"]

    # Update chat history for this user (keep last 20 messages to avoid unbounded growth)
    chat_history.append(HumanMessage(content=user_prompt))
    chat_history.append(AIMessage(content=agent_response))
    _user_chat_histories[telegram_id] = chat_history[-20:]

    return agent_response


# ==========================================
# 6. TERMINAL CHAT INTERFACE
# ==========================================
async def main_chat_loop():
    print(
        "🤖 Agent initialized with Semantic, Episodic, and Procedural memory orchestration."
    )
    print("Type 'exit' to quit.\n")

    chat_history = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Prepare the initial state
        initial_state: AgentState = {
            "user_prompt": user_input,
            "chat_history": chat_history,
            "semantic_context": "",
            "episodic_context": "",
            "procedural_context": "",
            "final_response": "",
            "skip_learning": False,
        }

        # Run the graph
        final_state = await app.ainvoke(initial_state)

        # Output the response
        agent_response = final_state["final_response"]
        print(f"\n🤖 Agent: {agent_response}")

        # Update chat history for the next turn
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=agent_response))


if __name__ == "__main__":
    # Run the async chat loop
    asyncio.run(main_chat_loop())
