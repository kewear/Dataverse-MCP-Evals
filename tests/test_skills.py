"""Tests for Skill scenarios (10-13)."""

import pytest

from eval.agent import Agent
from eval.evaluator import MCPEvaluator
from eval.state import StateManager
from tests.conftest import run_and_evaluate


# ── Scenario 10: Create skill ──


@pytest.mark.stage_setup
def test_create_skill(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 10: Create a new skill in Dataverse."""
    result = run_and_evaluate("create-skill", agent, evaluator, state)
    assert result.passed, _format_failure(result)


# ── Scenario 11: List all skills ──


@pytest.mark.stage_verify
def test_list_skills(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 11: List all skills in Dataverse."""
    result = run_and_evaluate("list-skills", agent, evaluator, state)
    assert result.passed, _format_failure(result)


# ── Scenario 12: Follow a skill ──


@pytest.mark.stage_verify
def test_follow_skill(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 12: Verify agent follows the skill instructions."""
    result = run_and_evaluate("follow-skill", agent, evaluator, state)
    assert result.passed, _format_failure(result)


# ── Scenario 13: Delete skill ──


@pytest.mark.stage_teardown
def test_delete_skill(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 13: Delete the created skill."""
    result = run_and_evaluate("delete-skill", agent, evaluator, state)
    assert result.passed, _format_failure(result)


def _format_failure(result) -> str:
    """Format a failure message with evaluation details."""
    lines = [f"Scenario '{result.scenario_name}' FAILED"]
    for score in result.scores:
        status = "✓" if score.passed else "✗"
        lines.append(f"  {status} {score.evaluator}: {score.score:.2f} — {score.reasoning[:200]}")
    if result.trace.final_response:
        lines.append(f"  Agent response: {result.trace.final_response[:300]}")
    return "\n".join(lines)
