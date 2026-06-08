# Unit tests for the tool/skill discovery system.
#
# Covers: frontmatter parsing, skill auto-discovery, skill loading,
# and the skill registry format.

import os
import sys

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ==========================================
# Frontmatter parsing (can test directly without full agent import)
# ==========================================
class TestFrontmatterParsing:
    """Tests for _parse_frontmatter in tools/__init__.py."""

    def test_parses_valid_frontmatter(self):
        from tools import _parse_frontmatter

        text = """---
name: composio
description: Use external apps through cloud tools.
triggers: github, gmail, calendar, slack
---

# Cloud Tools Skill

Instructions here.
"""
        meta = _parse_frontmatter(text)
        assert meta["name"] == "composio"
        assert meta["description"] == "Use external apps through cloud tools."
        assert "github" in meta["triggers"]

    def test_no_frontmatter_returns_empty(self):
        from tools import _parse_frontmatter

        text = "# Just a markdown file\n\nNo frontmatter here."
        meta = _parse_frontmatter(text)
        assert meta == {}

    def test_empty_values(self):
        from tools import _parse_frontmatter

        text = """---
name: minimal
description:
---
"""
        meta = _parse_frontmatter(text)
        assert meta["name"] == "minimal"
        assert meta["description"] == ""


# ==========================================
# Skill discovery integration
# ==========================================
class TestSkillDiscovery:
    """Tests for discover_skills scanning the tools/ directory."""

    def test_discovers_existing_skills(self):
        """Verify that the real tools/ directory has discoverable skills."""
        from tools import discover_skills

        skills = discover_skills()

        assert {"agent_identity_manager", "composio", "coding_agent"}.issubset(skills)
        assert "email" not in skills
        assert "calendar" not in skills
        assert "pdf_generator" not in skills

        # Check structure of any discovered skill
        for name, info in skills.items():
            assert "description" in info
            assert "skill_md_path" in info
            assert "tools" in info
            assert "tools_factory" in info
            assert isinstance(info["tools"], list)
            assert os.path.isfile(info["skill_md_path"])

    def test_skill_has_required_fields(self):
        """Each skill should have name, description, and path."""
        from tools import discover_skills

        skills = discover_skills()
        for name, info in skills.items():
            assert isinstance(name, str)
            assert len(name) > 0
            assert "description" in info


# ==========================================
# Skill loading (tool_manager)
# ==========================================
class TestSkillLoading:
    """Tests for load_skill and list_skills in tool_manager."""

    @pytest.fixture(autouse=True)
    def _patch_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")

    def test_list_skills_returns_dict(self):
        from tools.tool_manager import list_skills

        result = list_skills()
        assert isinstance(result, dict)

    def test_load_skill_returns_markdown_and_tools(self):
        from tools.tool_manager import list_skills, load_skill

        skills = list_skills()
        if not skills:
            pytest.skip("No skills discovered in tools/")

        first_skill_name = next(iter(skills))
        loaded = load_skill(first_skill_name)

        assert "markdown" in loaded
        assert "tools" in loaded
        assert isinstance(loaded["markdown"], str)
        assert len(loaded["markdown"]) > 0
        assert isinstance(loaded["tools"], list)

    def test_load_skill_invalid_name_raises(self):
        from tools.tool_manager import load_skill

        with pytest.raises(KeyError):
            load_skill("nonexistent_skill_that_does_not_exist")
