"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv

from eval.agent import Agent
from eval.evaluator import MCPEvaluator, compute_overall_pass
from eval.mcp_client import MCPClient
from eval.models import Scenario, ScenarioResult, Stage
from eval.state import StateManager

# Load environment variables
load_dotenv()

SCENARIOS_FILE = Path(__file__).parent.parent / "scenarios" / "scenarios.yaml"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _load_scenarios() -> list[Scenario]:
    """Load all scenarios from YAML."""
    with open(SCENARIOS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [Scenario(**s) for s in data["scenarios"]]


ALL_SCENARIOS = _load_scenarios()


def get_scenario(scenario_id: str) -> Scenario:
    """Get a scenario by ID."""
    for s in ALL_SCENARIOS:
        if s.id == scenario_id:
            return s
    raise ValueError(f"Scenario '{scenario_id}' not found")


@pytest.fixture(scope="session")
def mcp_client() -> MCPClient:
    """Shared MCP client for the test session."""
    server_url = os.environ.get("MCP_SERVER_URL", "https://bugbash02.crm10.dynamics.com/api/mcp")
    auth_token = os.environ.get("MCP_AUTH_TOKEN")

    client = MCPClient(server_url=server_url, auth_token=auth_token)
    client.initialize()
    client.discover_tools()
    yield client
    client.close()


@pytest.fixture(scope="session")
def agent(mcp_client: MCPClient) -> Agent:
    """Shared LLM agent for the test session."""
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    return Agent(
        mcp_client=mcp_client,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        deployment=deployment,
    )


@pytest.fixture(scope="session")
def evaluator() -> MCPEvaluator:
    """Shared evaluator for the test session."""
    conn_str = os.environ.get("AZURE_AI_CONNECTION_STRING")
    return MCPEvaluator(azure_ai_connection_string=conn_str)


@pytest.fixture(scope="session")
def state() -> StateManager:
    """Shared state manager for cross-stage resource tracking."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return StateManager(state_file=RESULTS_DIR / "state.json")


def run_and_evaluate(
    scenario_id: str,
    agent: Agent,
    evaluator: MCPEvaluator,
    state: StateManager,
) -> ScenarioResult:
    """Run a scenario through the agent and evaluate the result."""
    scenario = get_scenario(scenario_id)
    trace = agent.run(scenario.prompt)
    scores = evaluator.evaluate(scenario, trace)
    passed = compute_overall_pass(scores)

    result = ScenarioResult(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        stage=scenario.stage,
        trace=trace,
        scores=scores,
        passed=passed,
    )

    # Capture any resource IDs from tool call responses for state persistence
    for tc in trace.tool_calls:
        if tc.response and isinstance(tc.response, dict):
            # Look for record/resource IDs in the response
            for key in ("id", "recordId", "entityId", "tableId"):
                if key in tc.response:
                    state.set_resource(scenario.id, key, str(tc.response[key]))

        # Also store from nested content
        if tc.response and isinstance(tc.response, list):
            for item in tc.response:
                if isinstance(item, dict) and item.get("type") == "text":
                    state.set(f"{scenario.id}.response_text", item.get("text", ""))

    return result
