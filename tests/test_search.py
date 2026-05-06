"""Tests for Search & List scenarios (6-9)."""

import pytest

from eval.agent import Agent
from eval.evaluator import MCPEvaluator
from eval.state import StateManager
from tests.conftest import run_and_evaluate


# ── Scenario 6: List all tables ──


@pytest.mark.stage_verify
def test_list_tables(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 6: List all tables in Dataverse."""
    result = run_and_evaluate("list-tables", agent, evaluator, state)
    assert result.passed, _format_failure(result)


# ── Scenario 7: Search (search tool) ──


@pytest.mark.stage_verify
def test_search_generic(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 7: Search for something generic and verify search tool is used."""
    result = run_and_evaluate("search-generic", agent, evaluator, state)
    assert result.passed, _format_failure(result)


# ── Scenario 8: Search records (search_data tool) ──


@pytest.mark.stage_verify
def test_search_records(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 8: Search for specific records."""
    result = run_and_evaluate("search-records", agent, evaluator, state)
    assert result.passed, _format_failure(result)


# ── Scenario 9: Create table + search ──


@pytest.mark.stage_verify
def test_create_table_and_search(agent: Agent, evaluator: MCPEvaluator, state: StateManager):
    """Scenario 9: Verify previously created table appears in search."""
    result = run_and_evaluate("create-table-and-search", agent, evaluator, state)
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
