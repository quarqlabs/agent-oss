import os
import json
import asyncio
from typing import List, Dict
from urllib.request import urlretrieve

from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_core.messages import HumanMessage

from agent_connector import get_argus_response
from agent import wipe_all_memories
from langchain_openai import ChatOpenAI

# =========================================================
# CONFIG
# =========================================================

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
DATASET_DIR = os.path.join(ROOT_DIR, "eval_datasets")

CHECKPOINT_PATH = os.path.join(REPORTS_DIR, "eval_checkpoint.json")
RESULTS_PATH = os.path.join(REPORTS_DIR, "longmemeval_results.json")
DATASET_FILENAME = "longmemeval_s_cleaned.json"
DATASET_PATH = os.path.join(DATASET_DIR, DATASET_FILENAME)
DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    f"resolve/main/{DATASET_FILENAME}"
)

judge_llm = ChatOpenAI(
    model="gpt-5",
    api_key=SecretStr(api_key),
    temperature=0,
    reasoning_effort="medium"
)

QUESTION_IDS = []

# =========================================================
# CHECKPOINT HELPERS
# =========================================================


def get_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {"question_id": None, "last_chunk_index": -1}


def save_checkpoint(q_id, chunk_index):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"question_id": q_id, "last_chunk_index": chunk_index}, f)


def get_completed_question_ids():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            try:
                data = json.load(f)
                return [r["question_id"] for r in data]
            except Exception:
                return []
    return []


# =========================================================
# CORE FUNCTIONS
# =========================================================


def is_valid_dataset_file(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, list) and len(data) > 0
    except json.JSONDecodeError:
        return False


def ensure_dataset_local():
    """Download LongMemEval-S if the local cleaned dataset file is missing."""
    if is_valid_dataset_file(DATASET_PATH):
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    tmp_path = f"{DATASET_PATH}.tmp"

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"📥 LongMemEval-S dataset missing. Downloading to {DATASET_PATH}...")

    try:
        urlretrieve(DATASET_URL, tmp_path)

        if not is_valid_dataset_file(tmp_path):
            raise RuntimeError("Downloaded dataset is empty or invalid JSON.")

        os.replace(tmp_path, DATASET_PATH)
        print("✅ LongMemEval-S dataset downloaded successfully.")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"Failed to download LongMemEval-S from {DATASET_URL}"
        ) from e


def load_dataset_local():
    ensure_dataset_local()
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_history(
    sessions: List[List[Dict]],
    haystack_dates: List[str],
    haystack_session_ids: List[str],
    chunk_size: int = 8,
) -> List[Dict]:
    chunks = []

    for session_idx, session in enumerate(sessions):
        session_date = (
            haystack_dates[session_idx]
            if session_idx < len(haystack_dates)
            else ""
        )
        session_id = (
            haystack_session_ids[session_idx]
            if session_idx < len(haystack_session_ids)
            else f"session_{session_idx}"
        )

        current_chunk = []

        for msg in session:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            current_chunk.append(f"{role}: {content}")

            if len(current_chunk) >= chunk_size:
                chunks.append(
                    {
                        "text": "\n".join(current_chunk),
                        "date": session_date,
                        "session_id": session_id,
                        "session_idx": session_idx,
                    }
                )
                current_chunk = []

        if current_chunk:
            chunks.append(
                {
                    "text": "\n".join(current_chunk),
                    "date": session_date,
                    "session_id": session_id,
                    "session_idx": session_idx,
                }
            )

    return chunks


async def binary_judge(question: str, expected_answer: str, agent_answer: str) -> str:
    judge_prompt = f"""You are an expert evaluator grading an AI's memory recall and reasoning.

Evaluate whether the AGENT answer is semantically equivalent to the EXPECTED answer.

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

AGENT ANSWER:
{agent_answer}

EVALUATION RULES:

1. MEMORY FACT PRIORITY (CRITICAL):
The primary goal is to evaluate whether the AGENT correctly recalled and used the memory facts contained in the EXPECTED answer.

For memory questions involving names, dates, numbers, preferences, relationships, schedules, events, locations, purchases, plans, constraints, or historical facts:
- The AGENT must correctly recover the same underlying fact.
- Helpful advice, alternative suggestions, personalization, or generally reasonable responses do NOT count unless the underlying memory fact is correctly recalled.
- Missing the key memory fact is NO even if the answer is useful.

2. SEMANTIC EQUIVALENCE:
Output YES if the AGENT answer clearly expresses the same meaning as the EXPECTED answer, even if wording, phrasing, sentence structure, or level of detail differs.

3. FORMATTING TOLERANCE:
Do NOT penalize for:
- Full sentences
- Conversational tone
- First-person or third-person wording
- Bullet points
- Tables
- Markdown formatting
- Additional non-conflicting details

4. ADDITIONAL INFORMATION:
If the AGENT includes the correct answer and adds extra information that does not contradict the EXPECTED answer, output YES.

5. CONTRADICTION RULE (CRITICAL):
If the AGENT states a fact that directly contradicts the EXPECTED answer, output NO even if some surrounding information is correct.

A contradiction always overrides partial matches.

6. ABSTENTION RULE:
If the EXPECTED answer indicates that the information:
- was not mentioned,
- does not exist,
- is unavailable,
- cannot be determined,

then semantically equivalent responses such as:
- "not mentioned"
- "no evidence"
- "not specified"
- "I couldn't find that information"
- "there is no indication"

should be judged YES.

7. CONDITIONAL WORDING:
Conditional wording is acceptable only if the AGENT still clearly states the expected underlying fact.

Accept:
- "I believe it was June 5."
- "It appears the answer is June 5."

Reject:
- "Maybe June 5 or June 6."
- "Sometime in June."

8. NUMERIC ACCEPTANCE:
If the EXPECTED answer contains a numeric value:
- Accept equivalent expressions of the same value.
- Accept approximate wording ("about", "around", "roughly") ONLY when the same underlying number is stated.
- Do not accept different numbers.

8A. NUMERIC RANGE ACCEPTANCE (CRITICAL):

When the EXPECTED answer is a numeric value derived from
underlying quantities that may themselves be approximate,
uncertain, or expressed as ranges, accept an AGENT answer
that provides a mathematically consistent range or interval
instead of a single value, provided that:

- the EXPECTED value is consistent with the AGENT answer,
- the AGENT answer does not contradict the EXPECTED answer,
- the AGENT answer reflects the same underlying facts,
- the AGENT does not introduce unsupported numeric claims.

Do not reject an answer solely because it preserves uncertainty
that was already present in the underlying information.

9. DURATION GRANULARITY:
If the EXPECTED answer gives a duration in weeks, months, or years, accept an answer that gives the same duration plus smaller-unit detail, provided it does not contradict the expected value.

10. MISSING VARIABLE CASES:
If the EXPECTED answer states that a calculation cannot be completed because a specific value is missing, output YES if the AGENT correctly identifies the same missing value.

11. HALLUCINATION RULE:
Output NO if the AGENT invents unsupported facts that change the meaning of the EXPECTED answer.

12. FINAL DECISION:
Output YES only if the key memory fact(s) from the EXPECTED answer are correctly recalled and not contradicted.

Output NO if:
- the key memory fact is missing,
- the key memory fact is incorrect,
- the AGENT contradicts the EXPECTED answer,
- the AGENT gives a generic answer instead of recalling memory,
- the AGENT hallucinates unsupported facts.

Respond with exactly one token:
YES
or
NO
"""

    response = await judge_llm.ainvoke([HumanMessage(content=judge_prompt)])

    verdict = str(response.content).strip().upper()

    if verdict == "YES":
        return "YES"

    if verdict == "NO":
        return "NO"

    import re
    match = re.search(r"\b(YES|NO)\b", verdict)

    return match.group(1) if match else "NO"

async def feed_memory_chunks(
    chunks: List[Dict],
    user_id: str,
    q_id: str,
    resume_index: int,
):
    """Feeds chunks starting from the resume_index, using each chunk's haystack session date."""
    for i in range(resume_index + 1, len(chunks)):
        chunk = chunks[i]
        chunk_date = chunk.get("date", "")
        session_id = chunk.get("session_id", "")
        session_idx = chunk.get("session_idx", "")

        print(
            f"📦 Feeding chunk {i+1}/{len(chunks)} "
            f"| session={session_idx} "
            f"| session_id={session_id} "
            f"| date={chunk_date}"
        )

        prompt = f"Review and remember this conversation history:\n\n{chunk['text']}"

        await get_argus_response(
            user_prompt=prompt,
            chat_history=[],
            user_id=user_id,
            channel_type="benchmark",
            skip_learning=False,
            current_date=chunk_date,
        )

        save_checkpoint(q_id, i)
        await asyncio.sleep(1)


# =========================================================
# MAIN EVAL LOOP
# =========================================================


async def run_longmemeval():
    print("🚀 RUNNING LONGMEMEVAL")

    dataset = load_dataset_local()

    if QUESTION_IDS:
        dataset = [item for item in dataset if item["question_id"] in QUESTION_IDS]


    completed_ids = get_completed_question_ids()
    checkpoint = get_checkpoint()

    # Load existing results to append to them
    # Load existing results to append to them (SAFE VERSION)
    results = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                try:
                    results = json.loads(content)
                    print(f"📄 Loaded {len(results)} existing results.")
                except json.JSONDecodeError:
                    print(
                        f"⚠️ Warning: {RESULTS_PATH} is corrupted. Starting with empty results."
                    )
                    results = []
            else:
                results = []

    for index, item in enumerate(dataset):
        question_id = item["question_id"]
        question_date = item.get("question_date", "")

        # 1. Skip if question is already fully finished and judged
        if question_id in completed_ids:
            print(f"⏩ Question {question_id} already finished. Skipping.")
            continue

        print("\n" + "=" * 80)
        print(f"🧪 Processing Question {index+1}/{len(dataset)}: {question_id}")

        user_id = f"eval_user_{question_id}"

        haystack_sessions = item.get("haystack_sessions", [])
        haystack_dates = item.get("haystack_dates", [])
        haystack_session_ids = item.get("haystack_session_ids", [])

        chunks = chunk_history(
            haystack_sessions,
            haystack_dates,
            haystack_session_ids,
            chunk_size=8,
        )

        # 2. Determine where to start for chunks
        resume_chunk_idx = -1
        if checkpoint["question_id"] == question_id:
            resume_chunk_idx = checkpoint["last_chunk_index"]
            print(f"🔄 Resuming {question_id} from chunk {resume_chunk_idx + 1}")
        else:
            # 🛠️ WE ARE STARTING A BRAND NEW QUESTION. WIPE THE DB!
            print("🧽 New Question detected. Wiping Supabase clean...")
            # # ASK FOR CONFIRMATION
            while True:
                user_confirm = input("⚠️ Do you want to wipe the Supabase database clean for this question? (y/n): ").strip().lower()
                if user_confirm in ['y', 'yes']:
                    print("Wiping Supabase clean...")
                    await wipe_all_memories()
                    break
                elif user_confirm in ['n', 'no']:
                    print("⏭️ Skipping database wipe. Proceeding with existing memory...")
                    break
                else:
                    print("Invalid input. Please type 'y' or 'n'.")
            # await wipe_all_memories()

        # 3. Feed Chunks (if any are left)
        if resume_chunk_idx < len(chunks) - 1:
            await feed_memory_chunks(chunks, user_id, question_id, resume_chunk_idx)
        else:
            print(f"✅ All {len(chunks)} chunks already fed for this question.")

        # 4. Ask Final Question
        question = item["question"]
        expected_answer = item["answer"]
        question_type = item.get("question_type", "") or ""
        
        print(f"\n❓ ASKING: {question}")

        agent_answer, metrics, contexts = await get_argus_response(
            user_prompt=question,
            chat_history=[],
            user_id=user_id,
            channel_type="benchmark",
            skip_learning=True,
            current_date=question_date,
        )

        # 5. Judge
        verdict = await binary_judge(question, expected_answer, agent_answer)
        print(f"⚖️ VERDICT: {verdict}")

        # 6. Save Result and Clear Checkpoint for next question
        results.append(
            {
                "question_id": question_id,
                "question_type": question_type,
                "question": question,
                "expected_answer": expected_answer,
                "agent_answer": agent_answer,
                "result": verdict,
                "retrieved_context": {                     # 🛠️ NEW: Save retrieved memory
                    "semantic": contexts.get("semantic"),
                    "episodic": contexts.get("episodic"),
                    "procedural": contexts.get("procedural")
                }
            }
        )

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        # Reset checkpoint for the next question in the loop
        save_checkpoint(None, -1)

    print("\n🏆 EVALUATION COMPLETE. Report saved to reports/longmemeval_results.json")


if __name__ == "__main__":
    asyncio.run(run_longmemeval())
