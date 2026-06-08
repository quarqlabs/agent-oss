from coding_agents.approval import classify_action, classify_command, classify_file_write


def test_approval_policy_allows_safe_read_and_tests(tmp_path):
    assert classify_command("rg TODO .", tmp_path)["allow"] is True
    assert classify_command("python -m pytest tests", tmp_path)["allow"] is True
    assert classify_command("npm run build", tmp_path)["allow"] is True


def test_approval_policy_blocks_risky_commands(tmp_path):
    assert classify_command("rm -rf local_memory", tmp_path)["allow"] is False
    assert classify_command("git push origin main", tmp_path)["allow"] is False
    assert classify_command("pip install unknown-package", tmp_path)["allow"] is False
    assert classify_command("cat .env", tmp_path)["allow"] is False


def test_approval_policy_allows_workspace_edits_and_blocks_sensitive_or_outside(tmp_path):
    workspace_file = tmp_path / "agent.py"
    outside_file = tmp_path.parent / "outside.py"

    assert classify_file_write(str(workspace_file), tmp_path)["allow"] is True
    assert classify_file_write(str(outside_file), tmp_path)["allow"] is False
    assert classify_file_write(str(tmp_path / ".env"), tmp_path)["allow"] is False


def test_provider_required_confirmation_blocks(tmp_path):
    decision = classify_action(
        {"requires_user_confirmation": True, "command": "rg TODO ."},
        tmp_path,
    )
    assert decision["allow"] is False
    assert decision["risk"] == "provider_required"
