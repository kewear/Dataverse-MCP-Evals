"""Realistic evaluation runner — multi-turn conversations with postcondition validation."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import yaml

from eval.agent import Agent
from eval.evaluator import MCPEvaluator
from eval.mcp_client import MCPClient
from eval.models import (
    ConversationTrace,
    EvalScore,
    ResultStatus,
    ScenarioResult,
    Stage,
)

logger = logging.getLogger(__name__)

REALISTIC_FILE = Path(__file__).resolve().parent.parent.parent / "scenarios" / "realistic.yaml"


def _load_realistic_config() -> dict[str, Any]:
    with open(REALISTIC_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_realistic(
    agent: Agent,
    mcp: MCPClient,
    evaluator: MCPEvaluator,
    verbose: bool = False,
) -> list[ScenarioResult]:
    """Run realistic multi-turn conversation eval with postcondition checks."""
    config = _load_realistic_config()
    steps = config["steps"]

    click.echo(f"\n{'='*60}")
    click.echo("  REALISTIC EVALUATION (multi-turn conversation)")
    click.echo(f"{'='*60}\n")

    # Run steps one at a time (maintaining conversation state), with propagation waits
    click.echo(f"  Running {len(steps)} steps in a single conversation...\n")

    # We'll manage conversation manually to insert waits between steps
    messages: list[dict[str, Any]] = []
    results: list[ScenarioResult] = []

    for i, step in enumerate(steps):
        prompt = step["prompt"].strip()
        step_id = step["id"]

        # Check if previous step requested a propagation wait
        if i > 0:
            prev_wait = steps[i - 1].get("propagation_wait", 0)
            if prev_wait > 0:
                click.echo(f"  ⏳ Waiting {prev_wait}s for propagation...")
                time.sleep(prev_wait)

        click.echo(f"  ▸ Step {i+1}: {step_id}")

        # Run this step in the shared conversation
        step_traces = agent.run_conversation(
            [prompt],
            _existing_messages=messages,
        )
        trace = step_traces[0]

        # Run postcondition validator IMMEDIATELY after this step
        postcondition_result = _check_postcondition(step, mcp)

        # Evaluate success criteria (LLM-graded)
        criteria_score = _eval_criteria(step, trace, evaluator)

        # Determine pass/fail
        scores = [criteria_score]
        if postcondition_result is not None:
            scores.append(postcondition_result)

        passed = all(s.passed for s in scores)
        status = ResultStatus.PASSED if passed else ResultStatus.FAILED

        result = ScenarioResult(
            scenario_id=f"realistic-{step_id}",
            scenario_name=step_id,
            stage=Stage.VERIFY,
            status=status,
            trace=trace,
            scores=scores,
            passed=passed,
        )
        results.append(result)

        # Print result
        status_str = click.style("PASS", fg="green") if passed else click.style("FAIL", fg="red")
        click.echo(f"    {status_str}")
        for s in scores:
            mark = "✓" if s.passed else "✗"
            click.echo(f"      {mark} {s.evaluator}: {s.score:.2f}")

    # Print summary
    passed_count = sum(1 for r in results if r.passed)
    click.echo(f"\n{'='*60}")
    click.echo(f"  REALISTIC SUMMARY: {passed_count}/{len(results)} passed")
    click.echo(f"{'='*60}\n")

    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        color = "green" if r.passed else "red"
        click.echo(f"  {click.style(mark, fg=color)}  {r.scenario_name}")

    return results


def _check_postcondition(step: dict, mcp: MCPClient) -> EvalScore | None:
    """Run a direct MCP call to verify the actual state after agent action.

    Retries up to 3 times with 10s waits to handle propagation delays.
    """
    postcondition = step.get("postcondition")
    if postcondition is None:
        return None

    tool = postcondition["tool"]
    args = postcondition.get("args", {})
    expect_contains = postcondition.get("expect_contains", [])
    expect_empty = postcondition.get("expect_empty", False)
    expect_error = postcondition.get("expect_error", False)

    max_attempts = 3
    wait_between = 10  # seconds

    for attempt in range(max_attempts):
        result = _run_postcondition_check(
            mcp, tool, args, expect_contains, expect_empty, expect_error
        )
        if result.passed or attempt == max_attempts - 1:
            return result
        # Retry on failure (propagation delay)
        click.echo(f"      ⏳ Postcondition retry {attempt + 1}/{max_attempts - 1} (waiting {wait_between}s)...")
        time.sleep(wait_between)

    return result  # Should not reach here


def _run_postcondition_check(
    mcp: MCPClient,
    tool: str,
    args: dict,
    expect_contains: list[str],
    expect_empty: bool,
    expect_error: bool,
) -> EvalScore:
    """Single attempt at a postcondition check."""
    try:
        result = mcp.call_tool(tool, args)
        response_text = json.dumps(result.get("result", "")).lower()
        error_text = result.get("error", "") or ""

        if expect_error:
            # We expect the call to fail (e.g., table was deleted)
            if error_text or "not found" in response_text or "failed" in response_text:
                return EvalScore(
                    evaluator="postcondition",
                    score=1.0,
                    passed=True,
                    reasoning="Resource confirmed deleted (expected error received)",
                )
            else:
                return EvalScore(
                    evaluator="postcondition",
                    score=0.0,
                    passed=False,
                    reasoning=f"Expected resource to be gone but got: {response_text[:200]}",
                )

        if expect_empty:
            # Expect empty results (e.g., deleted record)
            if "[]" in response_text or response_text.strip() in ('""', '[]', 'null', ''):
                return EvalScore(
                    evaluator="postcondition",
                    score=1.0,
                    passed=True,
                    reasoning="Confirmed empty result (resource deleted)",
                )
            else:
                return EvalScore(
                    evaluator="postcondition",
                    score=0.0,
                    passed=False,
                    reasoning=f"Expected empty but got: {response_text[:200]}",
                )

        # Check for expected content
        if not expect_contains:
            return EvalScore(
                evaluator="postcondition",
                score=1.0,
                passed=True,
                reasoning="Postcondition call succeeded (no content check required)",
            )

        found = [t for t in expect_contains if t.lower() in response_text]
        missing = [t for t in expect_contains if t.lower() not in response_text]

        score = len(found) / len(expect_contains)
        passed = score >= 1.0

        details = [f"✓ Found: '{t}'" for t in found] + [f"✗ Missing: '{t}'" for t in missing]

        return EvalScore(
            evaluator="postcondition",
            score=score,
            passed=passed,
            reasoning="\n".join(details),
        )

    except Exception as e:
        if expect_error:
            return EvalScore(
                evaluator="postcondition",
                score=1.0,
                passed=True,
                reasoning=f"Resource confirmed deleted (exception: {e})",
            )
        return EvalScore(
            evaluator="postcondition",
            score=0.0,
            passed=False,
            reasoning=f"Postcondition check failed with error: {e}",
        )


def _eval_criteria(step: dict, trace: ConversationTrace, evaluator: MCPEvaluator) -> EvalScore:
    """Evaluate success criteria for a realistic step using the LLM grader."""
    criteria = step.get("success_criteria", [])
    if not criteria:
        return EvalScore(evaluator="success_criteria", score=1.0, passed=True, reasoning="No criteria")

    # Use the evaluator's LLM grading
    from eval.models import Scenario
    pseudo_scenario = Scenario(
        id=f"realistic-{step['id']}",
        name=step["id"],
        prompt=step["prompt"],
        stage=Stage.VERIFY,
        success_criteria=criteria,
    )
    scores = evaluator.evaluate(pseudo_scenario, trace)
    # Find the success_criteria score
    for s in scores:
        if s.evaluator == "success_criteria":
            return s

    return EvalScore(evaluator="success_criteria", score=1.0, passed=True, reasoning="No criteria evaluator")


# --- Propagation Measurement ---

def run_propagation_test(mcp: MCPClient, verbose: bool = False) -> dict[str, Any]:
    """Measure time-to-discoverable for a newly created table.

    This is NOT an agent test — it directly polls the MCP search endpoint
    to measure how long until a resource becomes searchable.
    """
    config = _load_realistic_config()
    prop_config = config.get("propagation_test")
    if not prop_config:
        click.echo("No propagation_test configured in realistic.yaml")
        return {}

    search_terms = prop_config["search_terms"]
    poll_interval = prop_config["poll_interval_seconds"]
    timeout = prop_config["timeout_seconds"]

    click.echo(f"\n{'='*60}")
    click.echo("  PROPAGATION TIMING MEASUREMENT")
    click.echo(f"{'='*60}\n")
    click.echo(f"  Polling search every {poll_interval}s for up to {timeout//60} minutes")
    click.echo(f"  Search terms: {search_terms}\n")

    results: dict[str, float | None] = {term: None for term in search_terms}
    start_time = time.time()

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time
        all_found = True

        for term in search_terms:
            if results[term] is not None:
                continue  # Already found

            try:
                response = mcp.call_tool("search", {"querytext": term})
                response_text = json.dumps(response.get("result", "")).lower()

                # Check if search returned meaningful results (not "no results")
                if "no results" not in response_text and term.lower() in response_text:
                    results[term] = elapsed
                    click.echo(f"  ✓ '{term}' found after {elapsed:.0f}s")
                else:
                    all_found = False
            except Exception:
                all_found = False

        if all_found:
            click.echo(f"\n  ✅ All terms discoverable after {elapsed:.0f}s")
            break

        remaining = timeout - elapsed
        click.echo(f"  ⏳ {elapsed:.0f}s elapsed, {remaining:.0f}s remaining...")
        time.sleep(poll_interval)
    else:
        click.echo(f"\n  ⏰ Timeout reached ({timeout}s)")

    # Summary
    click.echo(f"\n{'='*60}")
    click.echo("  PROPAGATION RESULTS")
    click.echo(f"{'='*60}\n")

    for term, elapsed_time in results.items():
        if elapsed_time is not None:
            click.echo(f"  '{term}': discoverable after {elapsed_time:.0f}s")
        else:
            click.echo(f"  '{term}': NOT discoverable within {timeout}s timeout")

    return {
        "search_terms": results,
        "poll_interval_seconds": poll_interval,
        "timeout_seconds": timeout,
        "timestamp": datetime.now(UTC).isoformat(),
    }
