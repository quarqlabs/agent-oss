"""
Connector module for external API routes (FastAPI, Webhooks, etc.) 
to interface with the Argus Agent smoothly.

This is the SINGLE integration gateway — all external callers (main.py,
webhooks, future APIs) go through this module. They never import from
agent.py or multimodal.py directly.
"""
from typing import Any, Awaitable, Callable, Optional, Sequence, Tuple

from langchain_core.messages import BaseMessage

from agent import app, AgentState


async def get_argus_response(
    user_prompt: str, 
    chat_history: Sequence[BaseMessage], 
    user_id: str, 
    channel_type: str,
    skip_learning: bool = False,
    current_date: Optional[str] = None,
    status_callback: Optional[
        Callable[[str, str, Optional[dict]], Awaitable[Any]]
    ] = None,
    attachments_context: str = "",
) -> Tuple[str, dict,dict]:
    """
    Public API to invoke the Argus Agent.
    
    Args:
        user_prompt (str): The current message from the user.
        chat_history (Sequence[BaseMessage]): Previous conversation context.
        user_id (str): A unique identifier for the user.
        channel_type (str): The platform (e.g., 'telegram', 'whatsapp').
        skip_learning (bool): Set to True to disable memory learning.
        
    Returns:
        Tuple[str, dict]: (Final response text, Metrics dictionary containing token usage)
    """
    
    # Construct the initial state required by the LangGraph application
    initial_state: AgentState = {
        "user_prompt": user_prompt,
        "chat_history": chat_history,
        "semantic_context": "",
        "episodic_context": "",
        "procedural_context": "",
        "selected_skills": [],
        "skill_markdown": "",
        "final_response": "",
        "skip_learning": skip_learning,
        "user_id": user_id,
        "channel_type": channel_type,
        "metrics": {}, # NEW: Initialize empty metrics bucket
        "current_date":current_date,
        "job_status_callback": status_callback,
        "attachments_context": attachments_context,
        "triage_agent_response": "",
        "triage_tool": False,
        "triage_retrieval": False,
    }

    try:
        # Run the full LangGraph pipeline (Retrieval -> Tools -> Generation -> Learning)
        final_state = await app.ainvoke(initial_state)

        # EXTRACT RETRIEVED MEMORY CONTEXT
        contexts = {
            "semantic": final_state.get("semantic_context", ""),
            "episodic": final_state.get("episodic_context", ""),
            "procedural": final_state.get("procedural_context", "")
        }
        
        return final_state["final_response"], final_state.get("metrics", {}), contexts
        
    except Exception as e:
        print(f"❌ [Error in Agent Connector]: {str(e)}")
        # Provide a graceful fallback response if the orchestration graph crashes
        return "I'm sorry, I encountered an internal error while processing your request. Please try again.", {}, {}
