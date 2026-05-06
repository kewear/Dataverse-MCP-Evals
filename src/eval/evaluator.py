"""Evaluation layer using Azure AI Foundry evaluators + custom MCP evaluators.

Azure AI Foundry SDK (azure-ai-evaluation) is optional. Install with:
    pip install -e ".[foundry]"
When not available, only custom evaluators are used.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .models import ConversationTrace, EvalScore, Scenario

logger = logging.getLogger(__name__)

# Optional Azure AI Foundry imports
try:
    from azure.ai.evaluation import ToolCallAccuracyEvaluator
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    HAS_FOUNDRY = True
except ImportError:
    HAS_FOUNDRY = False
    logger.info("Azure AI Foundry SDK not installed — using custom evaluators only")


class MCPEvaluator:
    """Evaluates agent execution traces against scenario criteria.

    Combines Azure AI Foundry built-in evaluators with custom
    MCP-specific evaluation logic.
    """

    def __init__(self, azure_ai_connection_string: str | None = None):
        self._foundry_client: AIProjectClient | None = None
        self._tool_call_evaluator: ToolCallAccuracyEvaluator | None = None

        if azure_ai_connection_string and HAS_FOUNDRY:
            try:
                credential = DefaultAzureCredential()
                self._foundry_client = AIProjectClient.from_connection_string(
                    conn_str=azure_ai_connection_string,
                    credential=credential,
                )
                self._tool_call_evaluator = ToolCallAccuracyEvaluator(
                    azure_ai_project=self._foundry_client,
                )
                logger.info("Azure AI Foundry evaluators initialized")
            except Exception as e:
                logger.warning("Could not initialize Foundry evaluators: %s", e)
                logger.info("Falling back to custom evaluators only")
        elif azure_ai_connection_string and not HAS_FOUNDRY:
            logger.warning(
                "AZURE_AI_CONNECTION_STRING provided but azure-ai-evaluation not installed. "
                "Install with: pip install -e '.[foundry]'"
            )

    def evaluate(self, scenario: Scenario, trace: ConversationTrace) -> list[EvalScore]:
        """Run all evaluators against a scenario trace."""
        scores: list[EvalScore] = []

        # 1. Tool call accuracy (custom — always available)
        scores.append(self._eval_tool_calls(scenario, trace))

        # 2. Tool parameter accuracy
        if scenario.expected_tool_params:
            scores.append(self._eval_tool_params(scenario, trace))

        # 3. Success criteria (custom text matching)
        scores.append(self._eval_success_criteria(scenario, trace))

        # 4. Azure AI Foundry ToolCallAccuracy (if available)
        if self._tool_call_evaluator and scenario.expected_tools:
            foundry_score = self._eval_foundry_tool_accuracy(scenario, trace)
            if foundry_score:
                scores.append(foundry_score)

        return scores

    def _eval_tool_calls(self, scenario: Scenario, trace: ConversationTrace) -> EvalScore:
        """Check that the expected MCP tools were called."""
        if not scenario.expected_tools:
            return EvalScore(
                evaluator="tool_call_check",
                score=1.0,
                passed=True,
                reasoning="No specific tools expected for this scenario",
            )

        called_tools = {tc.tool_name for tc in trace.tool_calls}
        expected = set(scenario.expected_tools)

        # Check if at least one expected tool was called
        matched = called_tools & expected
        missing = expected - called_tools

        if missing:
            return EvalScore(
                evaluator="tool_call_check",
                score=len(matched) / len(expected),
                passed=False,
                reasoning=f"Missing tool calls: {missing}. Called: {called_tools}",
            )

        return EvalScore(
            evaluator="tool_call_check",
            score=1.0,
            passed=True,
            reasoning=f"All expected tools called: {matched}",
        )

    def _eval_tool_params(self, scenario: Scenario, trace: ConversationTrace) -> EvalScore:
        """Check that tool call parameters match expectations."""
        params = scenario.expected_tool_params
        issues = []

        for tc in trace.tool_calls:
            args_str = json.dumps(tc.arguments).lower()

            # Check tablename_contains
            if "tablename_contains" in params:
                expected_name = params["tablename_contains"].lower()
                if expected_name in args_str:
                    continue  # Good

            # Check skill_name
            if "skill_name" in params:
                expected_name = params["skill_name"].lower()
                if expected_name in args_str:
                    continue

        # Check if any expected param was found in any tool call
        all_args = " ".join(json.dumps(tc.arguments).lower() for tc in trace.tool_calls)

        for key, value in params.items():
            if str(value).lower() not in all_args:
                issues.append(f"Expected param '{key}={value}' not found in tool calls")

        if issues:
            return EvalScore(
                evaluator="tool_param_check",
                score=0.0,
                passed=False,
                reasoning="; ".join(issues),
            )

        return EvalScore(
            evaluator="tool_param_check",
            score=1.0,
            passed=True,
            reasoning="All expected parameters found in tool calls",
        )

    def _eval_success_criteria(self, scenario: Scenario, trace: ConversationTrace) -> EvalScore:
        """Evaluate success criteria against the trace.

        This is a heuristic check — looks for evidence in the final response
        and tool call results that the criteria were met.
        """
        if not scenario.success_criteria:
            return EvalScore(
                evaluator="success_criteria",
                score=1.0,
                passed=True,
                reasoning="No success criteria defined",
            )

        # Build searchable text from the full trace
        searchable = trace.final_response.lower()
        for tc in trace.tool_calls:
            if tc.response:
                searchable += " " + json.dumps(tc.response).lower()
            searchable += " " + tc.tool_name.lower()

        # Check for key indicators
        met = 0
        total = len(scenario.success_criteria)
        details = []

        for criterion in scenario.success_criteria:
            criterion_lower = criterion.lower()

            # Extract key terms from the criterion for matching
            passed = False

            # Check if expected tool was called
            if "should call" in criterion_lower:
                for tool_name in scenario.expected_tools:
                    if tool_name in searchable:
                        passed = True
                        break
                if not scenario.expected_tools and trace.tool_calls:
                    passed = True

            # Check for confirmation language
            elif "confirm" in criterion_lower:
                confirm_words = ["created", "deleted", "removed", "success", "done", "complete"]
                passed = any(w in searchable for w in confirm_words)

            # Check for content presence
            elif "contain" in criterion_lower or "include" in criterion_lower or "should" in criterion_lower:
                # Look for quoted terms in the criterion
                import re
                quoted = re.findall(r"'([^']+)'", criterion)
                if quoted:
                    passed = any(q.lower() in searchable for q in quoted)
                else:
                    passed = True  # Generic criterion, assume met if we got a response

            # Check for returned data
            elif "return" in criterion_lower:
                passed = bool(trace.tool_calls and any(tc.response for tc in trace.tool_calls))

            else:
                # Default: check if we have any meaningful response
                passed = bool(trace.final_response.strip())

            if passed:
                met += 1
                details.append(f"✓ {criterion}")
            else:
                details.append(f"✗ {criterion}")

        score = met / total if total > 0 else 1.0
        return EvalScore(
            evaluator="success_criteria",
            score=score,
            passed=score >= 0.5,
            reasoning="\n".join(details),
        )

    def _eval_foundry_tool_accuracy(
        self, scenario: Scenario, trace: ConversationTrace
    ) -> EvalScore | None:
        """Use Azure AI Foundry ToolCallAccuracyEvaluator."""
        if not self._tool_call_evaluator:
            return None

        try:
            tool_calls = [
                {
                    "tool": tc.tool_name,
                    "params": tc.arguments,
                }
                for tc in trace.tool_calls
            ]
            expected = [{"tool": t} for t in scenario.expected_tools]

            result = self._tool_call_evaluator(
                query=scenario.prompt,
                tool_calls=tool_calls,
                expected_tool_calls=expected,
            )

            passed = result.get("tool_call_accuracy", 0) >= 0.5
            return EvalScore(
                evaluator="foundry_tool_call_accuracy",
                score=result.get("tool_call_accuracy", 0),
                passed=passed,
                reasoning=result.get("tool_call_accuracy_reason", ""),
            )
        except Exception as e:
            logger.warning("Foundry ToolCallAccuracy eval failed: %s", e)
            return EvalScore(
                evaluator="foundry_tool_call_accuracy",
                score=0,
                passed=False,
                reasoning=f"Evaluator error: {e}",
            )


def compute_overall_pass(scores: list[EvalScore]) -> bool:
    """A scenario passes if all critical evaluators pass."""
    critical = ["tool_call_check", "success_criteria"]
    for score in scores:
        if score.evaluator in critical and not score.passed:
            return False
    return True
