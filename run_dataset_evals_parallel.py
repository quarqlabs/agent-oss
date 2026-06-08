import os
import sys
import json
import glob
import asyncio
import subprocess
from typing import Dict, List
from urllib.request import urlretrieve

from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
DATASET_DIR = os.path.join(ROOT_DIR, "eval_datasets")

DATASET_FILENAME = "longmemeval_s_cleaned.json"
DATASET_PATH = os.path.join(DATASET_DIR, DATASET_FILENAME)
DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    f"resolve/main/{DATASET_FILENAME}"
)

MASTER_RESULTS_PATH = os.path.join(REPORTS_DIR, "longmemeval_results.json")

NUM_WORKERS = int(os.getenv("EVAL_WORKERS", "1"))
WORKER_ID = os.getenv("EVAL_WORKER_ID")
SHARD_PATH = os.getenv("EVAL_SHARD_PATH")
BASE_AGENT_ID = os.getenv("AGENT_ID") or "local_agent"

if WORKER_ID is not None:
    CHECKPOINT_PATH = os.path.join(
        REPORTS_DIR, f"eval_checkpoint.worker{WORKER_ID}.json"
    )
    RESULTS_PATH = os.path.join(
        REPORTS_DIR, f"longmemeval_results.worker{WORKER_ID}.json"
    )
else:
    CHECKPOINT_PATH = os.path.join(REPORTS_DIR, "eval_checkpoint.parallel.json")
    RESULTS_PATH = MASTER_RESULTS_PATH

judge_llm = ChatOpenAI(
    model="gpt-5",
    api_key=SecretStr(api_key),
    temperature=0,
    reasoning_effort="medium",
)

QUESTION_IDS = []

# These are intentionally imported lazily inside workers after AGENT_ID is set.
get_argus_response = None
wipe_all_memories = None


# =========================================================
# JSON / DATASET HELPERS
# =========================================================


def load_json_list(path: str) -> list:
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    try:
        data = json.loads(content)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        print(f"⚠️ Warning: {path} is corrupted. Treating it as empty.")
        return []


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    os.replace(tmp_path, path)


def upsert_result(path: str, row: dict):
    results = load_json_list(path)
    by_id = {
        r["question_id"]: r
        for r in results
        if isinstance(r, dict) and r.get("question_id")
    }
    by_id[row["question_id"]] = row
    save_json(path, list(by_id.values()))


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


def get_worker_result_paths() -> list[str]:
    return sorted(
        glob.glob(os.path.join(REPORTS_DIR, "longmemeval_results.worker*.json"))
    )


def get_all_completed_question_ids() -> set[str]:
    completed = set()

    for path in [MASTER_RESULTS_PATH, *get_worker_result_paths()]:
        for row in load_json_list(path):
            if isinstance(row, dict) and row.get("question_id"):
                completed.add(row["question_id"])

    return completed


def split_round_robin(items: list, num_workers: int) -> list[list]:
    return [items[i::num_workers] for i in range(num_workers)]


def merge_worker_results():
    merged = {}

    for row in load_json_list(MASTER_RESULTS_PATH):
        if isinstance(row, dict) and row.get("question_id"):
            merged[row["question_id"]] = row

    for path in get_worker_result_paths():
        for row in load_json_list(path):
            if isinstance(row, dict) and row.get("question_id"):
                merged[row["question_id"]] = row

    save_json(MASTER_RESULTS_PATH, list(merged.values()))
    print(f"✅ Merged {len(merged)} total results into {MASTER_RESULTS_PATH}")


# =========================================================
# CHECKPOINT HELPERS
# =========================================================


def get_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"question_id": None, "last_chunk_index": -1}


def save_checkpoint(q_id, chunk_index):
    save_json(CHECKPOINT_PATH, {"question_id": q_id, "last_chunk_index": chunk_index})


# =========================================================
# CORE EVAL FUNCTIONS
# =========================================================


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
            f"📦 [worker {WORKER_ID}] Feeding chunk {i+1}/{len(chunks)} "
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


async def run_worker_dataset(dataset: list):
    global get_argus_response, wipe_all_memories

    worker_label = WORKER_ID if WORKER_ID is not None else "single"
    agent_id = os.getenv("AGENT_ID") or f"{BASE_AGENT_ID}_eval_worker_{worker_label}"
    os.environ["AGENT_ID"] = agent_id

    from agent_connector import get_argus_response as _get_argus_response
    from agent import wipe_all_memories as _wipe_all_memories

    get_argus_response = _get_argus_response
    wipe_all_memories = _wipe_all_memories

    print(f"🚀 RUNNING LONGMEMEVAL worker={worker_label} agent_id={agent_id}")
    print(f"📄 Worker results: {RESULTS_PATH}")

    if QUESTION_IDS:
        dataset = [item for item in dataset if item["question_id"] in QUESTION_IDS]

    local_completed_ids = {
        row["question_id"]
        for row in load_json_list(RESULTS_PATH)
        if isinstance(row, dict) and row.get("question_id")
    }
    checkpoint = get_checkpoint()

    for index, item in enumerate(dataset):
        question_id = item["question_id"]
        question_date = item.get("question_date", "")

        if question_id in local_completed_ids:
            print(f"⏩ [worker {worker_label}] {question_id} already finished. Skipping.")
            continue

        print("\n" + "=" * 80)
        print(
            f"🧪 [worker {worker_label}] Processing "
            f"{index+1}/{len(dataset)}: {question_id}"
        )

        user_id = f"eval_user_worker_{worker_label}_{question_id}"

        chunks = chunk_history(
            item.get("haystack_sessions", []),
            item.get("haystack_dates", []),
            item.get("haystack_session_ids", []),
            chunk_size=8,
        )

        resume_chunk_idx = -1
        if checkpoint.get("question_id") == question_id:
            resume_chunk_idx = checkpoint.get("last_chunk_index", -1)
            print(f"🔄 [worker {worker_label}] Resuming from chunk {resume_chunk_idx + 1}")
        else:
            print(f"🧽 [worker {worker_label}] New question. Wiping local FAISS memory...")
            await wipe_all_memories()

        if resume_chunk_idx < len(chunks) - 1:
            await feed_memory_chunks(chunks, user_id, question_id, resume_chunk_idx)
        else:
            print(f"✅ [worker {worker_label}] All {len(chunks)} chunks already fed.")

        question = item["question"]
        expected_answer = item["answer"]
        question_type = item.get("question_type", "") or ""

        print(f"\n❓ [worker {worker_label}] ASKING: {question}")

        agent_answer, metrics, contexts = await get_argus_response(
            user_prompt=question,
            chat_history=[],
            user_id=user_id,
            channel_type="benchmark",
            skip_learning=True,
            current_date=question_date,
        )

        verdict = await binary_judge(question, expected_answer, agent_answer)
        print(f"⚖️ [worker {worker_label}] VERDICT: {verdict}")

        upsert_result(
            RESULTS_PATH,
            {
                "question_id": question_id,
                "question_type": question_type,
                "question": question,
                "expected_answer": expected_answer,
                "agent_answer": agent_answer,
                "result": verdict,
                "retrieved_context": {
                    "semantic": contexts.get("semantic"),
                    "episodic": contexts.get("episodic"),
                    "procedural": contexts.get("procedural"),
                },
            },
        )

        local_completed_ids.add(question_id)
        save_checkpoint(None, -1)
        checkpoint = get_checkpoint()

    print(f"\n🏁 Worker {worker_label} complete.")


# =========================================================
# MASTER / WORKER ENTRYPOINTS
# =========================================================


def run_parallel_evals():
    if NUM_WORKERS < 2:
        asyncio.run(run_worker_dataset(load_dataset_local()))
        return

    dataset = load_dataset_local()

    if QUESTION_IDS:
        dataset = [item for item in dataset if item["question_id"] in QUESTION_IDS]

    completed_ids = get_all_completed_question_ids()
    remaining = [item for item in dataset if item["question_id"] not in completed_ids]

    print("🚀 RUNNING PARALLEL LONGMEMEVAL")
    print(f"📄 Master results: {MASTER_RESULTS_PATH}")
    print(f"✅ Completed already: {len(completed_ids)}")
    print(f"🧪 Remaining: {len(remaining)}")
    print(f"⚙️ Workers: {NUM_WORKERS}")

    if not remaining:
        merge_worker_results()
        print("🏆 No remaining questions.")
        return

    shards = split_round_robin(remaining, NUM_WORKERS)
    procs = []

    for worker_id, shard in enumerate(shards):
        if not shard:
            continue

        shard_path = os.path.join(REPORTS_DIR, f"eval_shard.worker{worker_id}.json")
        save_json(shard_path, shard)

        env = os.environ.copy()
        env["EVAL_WORKER_ID"] = str(worker_id)
        env["EVAL_SHARD_PATH"] = shard_path
        env["AGENT_ID"] = f"{BASE_AGENT_ID}_eval_worker_{worker_id}"

        print(
            f"🚚 Starting worker {worker_id}: "
            f"{len(shard)} questions | AGENT_ID={env['AGENT_ID']}"
        )

        procs.append(subprocess.Popen([sys.executable, __file__], env=env))

    exit_codes = [p.wait() for p in procs]

    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more eval workers failed: {exit_codes}")

    merge_worker_results()
    print("\n🏆 PARALLEL EVALUATION COMPLETE.")


if __name__ == "__main__":
    if WORKER_ID is not None:
        if not SHARD_PATH:
            raise RuntimeError("Worker mode requires EVAL_SHARD_PATH.")

        with open(SHARD_PATH, "r", encoding="utf-8") as f:
            shard_dataset = json.load(f)

        asyncio.run(run_worker_dataset(shard_dataset))
    else:
        run_parallel_evals()
