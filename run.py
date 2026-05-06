"""CLI entry point for staged evaluation execution."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eval.agent import Agent
from eval.auth import get_dataverse_token, create_token_provider
from eval.evaluator import MCPEvaluator, compute_overall_pass
from eval.mcp_client import MCPClient
from eval.models import ResultStatus, Scenario, ScenarioResult, Stage
from eval.report import generate_html_report
from eval.state import StateManager

load_dotenv()

SCENARIOS_FILE = Path(__file__).parent / "scenarios" / "scenarios.yaml"
RESULTS_DIR = Path(__file__).parent / "results"


def _load_scenarios() -> list[Scenario]:
    with open(SCENARIOS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    scenarios = [Scenario(**s) for s in data["scenarios"]]
    _validate_dependency_graph(scenarios)
    return scenarios


def _validate_dependency_graph(scenarios: list[Scenario]) -> None:
    """Validate that all depends_on references exist and there are no cycles."""
    ids = {s.id for s in scenarios}
    for s in scenarios:
        if s.depends_on and s.depends_on not in ids:
            raise ValueError(f"Scenario '{s.id}' depends on unknown scenario '{s.depends_on}'")

    # Simple cycle detection via DFS
    deps = {s.id: s.depends_on for s in scenarios}
    for start in ids:
        visited: set[str] = set()
        current = start
        while current and current in deps:
            if current in visited:
                raise ValueError(f"Circular dependency detected involving '{current}'")
            visited.add(current)
            current = deps.get(current)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy/sensitive loggers even in verbose mode
    for noisy in ("msal", "urllib3", "httpcore", "azure.identity", "azure.core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _create_components() -> tuple[MCPClient, Agent, MCPEvaluator, StateManager]:
    # Azure AI Foundry (LLM inference via v1 endpoint)
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    # Dataverse MCP Server (may be in a different tenant)
    server_url = os.environ.get("MCP_SERVER_URL")
    if not server_url:
        raise SystemExit("MCP_SERVER_URL is required in .env (e.g. https://myorg.crm.dynamics.com/api/mcp)")
    auth_token = os.environ.get("MCP_AUTH_TOKEN")

    # If no static token, create a token provider that auto-refreshes
    token_provider = None
    if not auth_token:
        org_url = server_url.split("/api/mcp")[0]
        tenant_id = os.environ.get("MCP_TENANT_ID")
        token_provider = create_token_provider(org_url=org_url, tenant_id=tenant_id)

    # Azure AI Foundry evaluation (optional)
    conn_str = os.environ.get("AZURE_AI_CONNECTION_STRING")

    mcp = MCPClient(server_url=server_url, auth_token=auth_token, token_provider=token_provider)
    mcp.initialize()
    mcp.discover_tools()

    agent = Agent(
        mcp_client=mcp,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        deployment=deployment,
    )
    evaluator = MCPEvaluator(azure_ai_connection_string=conn_str)
    state = StateManager(state_file=RESULTS_DIR / "state.json")

    return mcp, agent, evaluator, state


def _resolve_dependencies(
    scenario: Scenario,
    all_scenarios: list[Scenario],
    status_map: dict[str, ResultStatus],
) -> list[Scenario]:
    """Walk the dependency chain and return unmet prerequisites in order.

    Returns a list of scenarios that must run before `scenario`, ordered
    so that each dependency comes before any scenario that depends on it.
    """
    if not scenario.depends_on:
        return []

    by_id = {s.id: s for s in all_scenarios}
    chain: list[Scenario] = []
    visited: set[str] = set()

    def _walk(dep_id: str) -> None:
        if dep_id in visited:
            return
        visited.add(dep_id)
        dep = by_id.get(dep_id)
        if dep is None:
            return
        # Already passed — no need to re-run
        if status_map.get(dep_id) == ResultStatus.PASSED:
            return
        # Recurse into *its* dependencies first
        if dep.depends_on:
            _walk(dep.depends_on)
        chain.append(dep)

    _walk(scenario.depends_on)
    return chain


def _run_single_scenario(
    scenario: Scenario,
    agent: Agent,
    evaluator: MCPEvaluator,
    state: StateManager,
    status_map: dict[str, ResultStatus],
    *,
    is_dep: bool = False,
) -> ScenarioResult:
    """Execute a single scenario, evaluate it, and persist status."""
    prefix = "  ↳ dep" if is_dep else "  ▸"
    click.echo(f"{prefix} {scenario.name} ({scenario.id})")

    try:
        trace = agent.run(scenario.prompt)
        scores = evaluator.evaluate(scenario, trace)
        passed = compute_overall_pass(scores)
        result_status = ResultStatus.PASSED if passed else ResultStatus.FAILED

        result = ScenarioResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            stage=scenario.stage,
            status=result_status,
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
            status=ResultStatus.ERROR,
            passed=False,
            error=str(e),
        )
        click.echo(f"    {click.style('ERROR', fg='red')}: {e}")

    # Track status for downstream dependencies
    status_map[scenario.id] = result.status
    state.set(f"scenario.{scenario.id}.status", result.status.value)
    return result


def _check_dependency(
    scenario: Scenario,
    status_map: dict[str, ResultStatus],
    stage: Stage,
) -> str | None:
    """Check if a scenario's dependency has been met.

    Returns None if OK to run, or a skip reason string.
    Teardown scenarios run best-effort even if dependencies failed.
    """
    if not scenario.depends_on:
        return None

    dep_status = status_map.get(scenario.depends_on)

    if dep_status is None:
        if stage == Stage.TEARDOWN:
            return None
        return f"Dependency '{scenario.depends_on}' has not been executed"

    if dep_status == ResultStatus.PASSED:
        return None

    if stage == Stage.TEARDOWN:
        return None

    return f"Dependency '{scenario.depends_on}' has status '{dep_status.value}'"


# Propagation wait between auto-run setup deps and dependent scenarios
_PROPAGATION_WAIT = 30  # seconds
_PROPAGATION_RETRIES = 5  # total attempts


def _run_scenarios(
    scenarios: list[Scenario],
    stage: Stage,
    agent: Agent,
    evaluator: MCPEvaluator,
    state: StateManager,
    status_map: dict[str, ResultStatus] | None = None,
) -> list[ScenarioResult]:
    """Run all scenarios for a given stage, auto-running missing dependencies."""
    if status_map is None:
        # Load persisted status from state if not provided (separate stage runs)
        status_map = {}
        for s in scenarios:
            persisted = state.get(f"scenario.{s.id}.status")
            if persisted:
                status_map[s.id] = ResultStatus(persisted)

    stage_scenarios = [s for s in scenarios if s.stage == stage]
    results: list[ScenarioResult] = []
    already_ran: set[str] = set()

    click.echo(f"\n{'='*60}")
    click.echo(f"  Stage: {stage.value.upper()} ({len(stage_scenarios)} scenarios)")
    click.echo(f"{'='*60}\n")

    for scenario in stage_scenarios:
        # Auto-run any unmet dependencies from earlier stages
        deps = _resolve_dependencies(scenario, scenarios, status_map)
        ran_deps = False
        for dep in deps:
            if dep.id in already_ran:
                continue
            already_ran.add(dep.id)
            ran_deps = True
            dep_result = _run_single_scenario(dep, agent, evaluator, state, status_map, is_dep=True)
            results.append(dep_result)

        # Now check if the dependency chain passed
        skip_reason = _check_dependency(scenario, status_map, stage)
        if skip_reason:
            click.echo(f"  ▸ {scenario.name} ({scenario.id})")
            result = ScenarioResult(
                scenario_id=scenario.id,
                scenario_name=scenario.name,
                stage=scenario.stage,
                status=ResultStatus.SKIPPED,
                passed=False,
                error=skip_reason,
            )
            status_map[scenario.id] = ResultStatus.SKIPPED
            state.set(f"scenario.{scenario.id}.status", ResultStatus.SKIPPED.value)
            click.echo(f"    {click.style('SKIP', fg='yellow')}: {skip_reason}")
            results.append(result)
            continue

        # Use propagation retry for any scenario that has dependencies
        # (cross-stage auto-deps OR same-stage deps like create-record → create-table)
        has_deps = ran_deps or scenario.depends_on
        if has_deps:
            result = _run_with_propagation_retry(
                scenario, agent, evaluator, state, status_map
            )
        else:
            result = _run_single_scenario(scenario, agent, evaluator, state, status_map)

        already_ran.add(scenario.id)
        results.append(result)

    return results


def _run_with_propagation_retry(
    scenario: Scenario,
    agent: Agent,
    evaluator: MCPEvaluator,
    state: StateManager,
    status_map: dict[str, ResultStatus],
) -> ScenarioResult:
    """Run a scenario with propagation retry — try first, then wait+retry on failure."""
    result = _run_single_scenario(scenario, agent, evaluator, state, status_map)

    if result.status == ResultStatus.PASSED:
        return result

    # First attempt failed — retry with waits for Dataverse propagation
    for attempt in range(2, _PROPAGATION_RETRIES + 1):
        click.echo(f"\n  ⏳ Waiting {_PROPAGATION_WAIT}s for propagation (attempt {attempt}/{_PROPAGATION_RETRIES})...")
        time.sleep(_PROPAGATION_WAIT)

        result = _run_single_scenario(scenario, agent, evaluator, state, status_map)

        if result.status == ResultStatus.PASSED:
            return result

        if attempt < _PROPAGATION_RETRIES:
            click.echo(f"    ↻ Will retry after another {_PROPAGATION_WAIT}s wait...")

    return result  # Return last attempt's result


def _save_results(results: list[ScenarioResult]) -> Path:
    """Save combined HTML report with embedded JSON (single shareable file)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    data = [r.model_dump(mode="json") for r in results]
    json_str = json.dumps(data, indent=2, default=str)

    html_file = RESULTS_DIR / f"report_{timestamp}.html"
    generate_html_report(results, html_file, json_data=json_str)

    return html_file


def _print_summary(results: list[ScenarioResult]) -> None:
    """Print a summary table of results."""
    passed = sum(1 for r in results if r.status == ResultStatus.PASSED)
    failed = sum(1 for r in results if r.status == ResultStatus.FAILED)
    skipped = sum(1 for r in results if r.status == ResultStatus.SKIPPED)
    errored = sum(1 for r in results if r.status == ResultStatus.ERROR)

    click.echo(f"\n{'='*60}")
    parts = [f"{passed} passed", f"{failed} failed"]
    if skipped:
        parts.append(f"{skipped} skipped")
    if errored:
        parts.append(f"{errored} errors")
    click.echo(f"  SUMMARY: {', '.join(parts)} — {len(results)} total")
    click.echo(f"{'='*60}\n")

    for r in results:
        if r.status == ResultStatus.PASSED:
            status = click.style("PASS", fg="green")
        elif r.status == ResultStatus.SKIPPED:
            status = click.style("SKIP", fg="yellow")
        elif r.status == ResultStatus.ERROR:
            status = click.style("ERR ", fg="red")
        else:
            status = click.style("FAIL", fg="red")
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
        html_file = _save_results(results)
        _print_summary(results)

        click.echo(f"Report:  {html_file}")
    finally:
        mcp.close()


@cli.command()
def verify():
    """Run verify stage — validate resources exist."""
    scenarios = _load_scenarios()
    mcp, agent, evaluator, state = _create_components()
    try:
        results = _run_scenarios(scenarios, Stage.VERIFY, agent, evaluator, state)
        html_file = _save_results(results)
        _print_summary(results)

        click.echo(f"Report:  {html_file}")
    finally:
        mcp.close()


@cli.command()
def teardown():
    """Run teardown stage — clean up resources."""
    scenarios = _load_scenarios()
    mcp, agent, evaluator, state = _create_components()
    try:
        results = _run_scenarios(scenarios, Stage.TEARDOWN, agent, evaluator, state)
        html_file = _save_results(results)
        _print_summary(results)

        click.echo(f"Report:  {html_file}")
    finally:
        mcp.close()


@cli.command(name="all")
@click.option("--wait", default=900, help="Seconds to wait between setup and verify (default: 900)")
def run_all(wait: int):
    """Run all stages: setup → wait → verify → teardown."""
    scenarios = _load_scenarios()
    mcp, agent, evaluator, state = _create_components()
    all_results: list[ScenarioResult] = []
    status_map: dict[str, ResultStatus] = {}

    try:
        # Setup
        results = _run_scenarios(scenarios, Stage.SETUP, agent, evaluator, state, status_map)
        all_results.extend(results)

        # Wait for propagation
        click.echo(f"\n⏳ Waiting {wait} seconds for Dataverse propagation...")
        for remaining in range(wait, 0, -60):
            click.echo(f"   {remaining}s remaining...")
            time.sleep(min(60, remaining))

        # Verify
        results = _run_scenarios(scenarios, Stage.VERIFY, agent, evaluator, state, status_map)
        all_results.extend(results)

        # Teardown (best-effort — runs even if verify failed)
        results = _run_scenarios(scenarios, Stage.TEARDOWN, agent, evaluator, state, status_map)
        all_results.extend(results)

        html_file = _save_results(all_results)
        _print_summary(all_results)

        click.echo(f"Report:  {html_file}")

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


@cli.command()
def realistic():
    """Run realistic evaluation (multi-turn conversation, natural language, postconditions)."""
    _setup_logging(False)
    mcp, agent, evaluator, state = _create_components()

    from eval.realistic import run_realistic
    results = run_realistic(agent, mcp, evaluator)

    html_file = _save_results(results)

    click.echo(f"Report:  {html_file}")


@cli.command()
def propagation():
    """Measure propagation timing (how long until a new table is searchable)."""
    _setup_logging(False)
    mcp, agent, evaluator, state = _create_components()

    from eval.realistic import run_propagation_test
    results = run_propagation_test(mcp)

    # Save propagation results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"propagation_{timestamp}.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    click.echo(f"\nResults saved: {out_file}")


def main():
    cli()


if __name__ == "__main__":
    main()
