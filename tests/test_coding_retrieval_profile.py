from agent import coding_retrieval_profile, is_coding_task_prompt


def test_coding_prompt_detection():
    assert is_coding_task_prompt("please implement the new CLI command")
    assert is_coding_task_prompt("fix bug in the repo")
    assert not is_coding_task_prompt("what did I buy last week?")


def test_coding_retrieval_profile_is_shallow():
    profile = coding_retrieval_profile("implement the coding agent tests")

    assert profile["search_mode"] == "standard"
    assert profile["top_k"] <= 4
    assert profile["keyword_top_k"] <= 4
    assert profile["max_lines"] <= 24
    assert profile["threshold"] >= 0.38
