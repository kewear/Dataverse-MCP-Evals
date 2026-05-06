"""Tests for CRUD scenarios (1-5): Create/Fetch/Delete table and records."""

import pytest

from eval.agent import Agent
from eval.evaluator import MCPEvaluator
from eval.state import StateManager
from tests.conftest import run_and_evaluate


# ── Scenario 1: Create table ──


@pytest.mark.stage_setup
def test_create_table(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 1: Create a new table in Dataverse."""
    result = run_and_evaluate("create-table", agent, evaluator, state)
    assert result.passed, _format_failure(result)


# ── Scenario 2: Create record ──


@pytest.mark.stage_setup
def test_create_record(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 2: Create a record in the table."""
    result = run_and_evaluate("create-record", agent, evaluator, state)
    assert result.passed, _format_failure(result)


# ── Scenario 3: Fetch record ──


@pytest.mark.stage_verify
def test_fetch_record(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 3: Fetch details of the created record."""
    result = run_and_evaluate("fetch-record", agent, evaluator, state)
    assert result.passed, _format_failure(result)


# ── Scenario 4: Delete record ──


@pytest.mark.stage_teardown
def test_delete_record(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 4: Delete the record."""
    result = run_and_evaluate("delete-record", agent, evaluator, state)
    assert result.passed, _format_failure(result)


# ── Scenario 5: Delete table ──


@pytest.mark.stage_teardown
def test_delete_table(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 5: Delete the table."""
    result = run_and_evaluate("delete-table", agent, evaluator, state)
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
