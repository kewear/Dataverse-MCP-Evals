"""CLI entry point for staged evaluation execution."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eval.agent import Agent
from eval.evaluator import MCPEvaluator, compute_overall_pass
from eval.mcp_client import MCPClient
from eval.models import Scenario, ScenarioResult, Stage
from eval.state import StateManager

load_dotenv()

SCENARIOS_FILE = Path(__file__).parent / "scenarios" / "scenarios.yaml"
RESULTS_DIR = Path(__file__).parent / "results"


def _load_scenarios() -> list[Scenario]:
    with open(SCENARIOS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [Scenario(**s) for s in data["scenarios"]]


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _create_components() -> tuple[MCPClient, Agent, MCPEvaluator, StateManager]:
    server_url = os.environ.get("MCP_SERVER_URL", "https://bugbash02.crm10.dynamics.com/api/mcp")
    auth_token = os.environ.get("MCP_AUTH_TOKEN")
    github_token = os.environ["GITHUB_TOKEN"]
    model = os.environ.get("GITHUB_MODELS_MODEL", "gpt-4o")
    conn_str = os.environ.get("AZURE_AI_CONNECTION_STRING")

    mcp = MCPClient(server_url=server_url, auth_token=auth_token)
    mcp.initialize()
    mcp.discover_tools()

    agent = Agent(mcp_client=mcp, github_token=github_token, model=model)
    evaluator = MCPEvaluator(azure_ai_connection_string=conn_str)
    state = StateManager(state_file=RESULTS_DIR / "state.json")

    return mcp, agent, evaluator, state


def _run_scenarios(
    scenarios: list[Scenario],
    stage: Stage,
    agent: Agent,
    evaluator: MCPEvaluator,
    state: StateManager,
) -> list[ScenarioResult]:
    """Run all scenarios for a given stage."""
    stage_scenarios = [s for s in scenarios if s.stage == stage]
    results: list[ScenarioResult] = []

    click.echo(f"\n{'='*60}")
    click.echo(f"  Stage: {stage.value.upper()} ({len(stage_scenarios)} scenarios)")
    click.echo(f"{'='*60}\n")

    for scenario in stage_scenarios:
        click.echo(f"  ▸ {scenario.name} ({scenario.id})")
        try:
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

            # Persist resource state
            for tc in trace.tool_calls:
                if tc.response and isinstance(tc.response, dict):
                    for key in ("id", "recordId", "entityId"):
                        if key in tc.response:
                            state.set_resource(scenario.id, key, str(tc.response[key]))
                if tc.response and isinstance(tc.response, list):
                    for item in tc.response:
                        if isinstance(item, dict) and item.get("type") == "text":
                            state.set(f"{scenario.id}.response_text", item.get("text", ""))

            status = click.style("PASS", fg="green") if passed else click.style("FAIL", fg="red")
            click.echo(f"    {status}")
            for score in scores:
                icon = "✓" if score.passed else "✗"
                click.echo(f"      {icon} {score.evaluator}: {score.score:.2f}")

        except Exception as e:
            result = ScenarioResult(
                scenario_id=scenario.id,
                scenario_name=scenario.name,
                stage=scenario.stage,
                trace=trace if "trace" in dir() else None,
                passed=False,
                error=str(e),
            )
            click.echo(f"    {click.style('ERROR', fg='red')}: {e}")

        results.append(result)

    return results


def _save_results(results: list[ScenarioResult]) -> Path:
    """Save results to a timestamped JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"results_{timestamp}.json"

    data = [r.model_dump(mode="json") for r in results]
    output_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return output_file


def _print_summary(results: list[ScenarioResult]) -> None:
    """Print a summary table of results."""
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    click.echo(f"\n{'='*60}")
    click.echo(f"  SUMMARY: {passed} passed, {failed} failed, {len(results)} total")
    click.echo(f"{'='*60}\n")

    for r in results:
        status = click.style("PASS", fg="green") if r.passed else click.style("FAIL", fg="red")
        click.echo(f"  {status}  {r.scenario_name} ({r.scenario_id})")

    click.echo()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    """Dataverse MCP Server Evaluation Framework."""
    _setup_logging(verbose)


@cli.command()
def setup():
    """Run setup stage — create resources."""
    scenarios = _load_scenarios()
    mcp, agent, evaluator, state = _create_components()
    try:
        results = _run_scenarios(scenarios, Stage.SETUP, agent, evaluator, state)
        output = _save_results(results)
        _print_summary(results)
        click.echo(f"Results saved to: {output}")
    finally:
        mcp.close()


@cli.command()
def verify():
    """Run verify stage — validate resources exist."""
    scenarios = _load_scenarios()
    mcp, agent, evaluator, state = _create_components()
    try:
        results = _run_scenarios(scenarios, Stage.VERIFY, agent, evaluator, state)
        output = _save_results(results)
        _print_summary(results)
        click.echo(f"Results saved to: {output}")
    finally:
        mcp.close()


@cli.command()
def teardown():
    """Run teardown stage — clean up resources."""
    scenarios = _load_scenarios()
    mcp, agent, evaluator, state = _create_components()
    try:
        results = _run_scenarios(scenarios, Stage.TEARDOWN, agent, evaluator, state)
        output = _save_results(results)
        _print_summary(results)
        click.echo(f"Results saved to: {output}")
    finally:
        mcp.close()


@cli.command(name="all")
@click.option("--wait", default=900, help="Seconds to wait between setup and verify (default: 900)")
def run_all(wait: int):
    """Run all stages: setup → wait → verify → teardown."""
    scenarios = _load_scenarios()
    mcp, agent, evaluator, state = _create_components()
    all_results: list[ScenarioResult] = []

    try:
        # Setup
        results = _run_scenarios(scenarios, Stage.SETUP, agent, evaluator, state)
        all_results.extend(results)

        # Wait for propagation
        click.echo(f"\n⏳ Waiting {wait} seconds for Dataverse propagation...")
        for remaining in range(wait, 0, -60):
            click.echo(f"   {remaining}s remaining...")
            time.sleep(min(60, remaining))

        # Verify
        results = _run_scenarios(scenarios, Stage.VERIFY, agent, evaluator, state)
        all_results.extend(results)

        # Teardown
        results = _run_scenarios(scenarios, Stage.TEARDOWN, agent, evaluator, state)
        all_results.extend(results)

        output = _save_results(all_results)
        _print_summary(all_results)
        click.echo(f"Results saved to: {output}")

    finally:
        mcp.close()


@cli.command()
def status():
    """Show current state (persisted resources from previous runs)."""
    state = StateManager(state_file=RESULTS_DIR / "state.json")
    data = state.get_all()
    if not data:
        click.echo("No state found. Run 'setup' first.")
        return
    click.echo("Current state:")
    for key, value in data.items():
        click.echo(f"  {key}: {value}")


def main():
    cli()


if __name__ == "__main__":
    main()
