"""Evaluation layer using Azure AI Foundry evaluators + custom MCP evaluators.

Azure AI Foundry SDK (azure-ai-evaluation) is optional. Install with:
    pip install -e ".[foundry]"
When not available, only custom evaluators are used.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .models import ConversationTrace, EvalScore, Scenario

logger = logging.getLogger(__name__)

# Minimum scores required for each critical evaluator to pass
CRITICAL_THRESHOLDS: dict[str, float] = {
    "tool_call_check": 0.7,
    "tool_param_check": 0.7,
    "success_criteria": 0.7,
    "response_content": 1.0,  # All expected strings must be present
}

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

        # 4. Response content validation (concrete string checks against tool responses)
        if scenario.expected_response_contains:
            scores.append(self._eval_response_content(scenario, trace))

        # 5. Azure AI Foundry ToolCallAccuracy (if available)
        if self._tool_call_evaluator and scenario.expected_tools:
            foundry_score = self._eval_foundry_tool_accuracy(scenario, trace)
            if foundry_score:
                scores.append(foundry_score)

        return scores

    def _eval_tool_calls(self, scenario: Scenario, trace: ConversationTrace) -> EvalScore:
        """Check that the expected MCP tools were called.

        A scenario passes if ANY of the expected_tools OR acceptable_tools were called.
        expected_tools are the ideal tools; acceptable_tools are valid alternatives.
        """
        if not scenario.expected_tools and not scenario.acceptable_tools:
            return EvalScore(
                evaluator="tool_call_check",
                score=1.0,
                passed=True,
                reasoning="No specific tools expected for this scenario",
            )

        called_tools = {tc.tool_name for tc in trace.tool_calls}
        expected = set(scenario.expected_tools)
        acceptable = set(scenario.acceptable_tools)
        all_valid = expected | acceptable

        matched = called_tools & all_valid
        matched_expected = called_tools & expected

        if not matched:
            return EvalScore(
                evaluator="tool_call_check",
                score=0.0,
                passed=False,
                reasoning=f"No valid tools called. Expected: {expected}. Acceptable: {acceptable}. Called: {called_tools}",
            )

        # Full score if an expected tool was used, partial if only acceptable
        if matched_expected:
            return EvalScore(
                evaluator="tool_call_check",
                score=1.0,
                passed=True,
                reasoning=f"Expected tools called: {matched_expected}",
            )

        return EvalScore(
            evaluator="tool_call_check",
            score=0.8,
            passed=True,
            reasoning=f"Acceptable alternative tools used: {matched & acceptable}. Ideal: {expected}",
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

    def _eval_response_content(self, scenario: Scenario, trace: ConversationTrace) -> EvalScore:
        """Validate that expected strings appear in tool responses (not just the LLM summary).

        This is a concrete, verifiable check — each string in expected_response_contains
        must appear (case-insensitive) in the actual tool response data.
        Also fails if any tool response indicates an execution error.
        """
        expected = scenario.expected_response_contains
        if not expected:
            return EvalScore(
                evaluator="response_content",
                score=1.0,
                passed=True,
                reasoning="No expected response content defined",
            )

        # Build searchable text from tool responses only (not the LLM's final answer)
        # Use only successful responses (skip errored ones where the agent self-corrected)
        tool_response_text = ""
        for tc in trace.tool_calls:
            if tc.response is not None:
                resp_str = _flatten_response(tc.response)
                # Skip responses that are clearly errors (agent may retry)
                if "tool execution failed" in resp_str.lower() and isinstance(tc.response, dict) and tc.response.get("isError"):
                    continue
                tool_response_text += " " + resp_str
        tool_response_text_lower = tool_response_text.lower()

        # If ALL tool responses were errors, use the full text for matching (will likely fail)
        if not tool_response_text.strip():
            for tc in trace.tool_calls:
                if tc.response is not None:
                    tool_response_text += " " + _flatten_response(tc.response)
            tool_response_text_lower = tool_response_text.lower()

        found = []
        missing = []
        for term in expected:
            if term.lower() in tool_response_text_lower:
                found.append(term)
            else:
                missing.append(term)

        score = len(found) / len(expected) if expected else 1.0
        details = [f"✓ Found: '{t}'" for t in found] + [f"✗ Missing: '{t}'" for t in missing]

        return EvalScore(
            evaluator="response_content",
            score=score,
            passed=score >= CRITICAL_THRESHOLDS.get("response_content", 1.0),
            reasoning="\n".join(details),
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

        # Combine expected + acceptable tools for "should call" checks
        all_valid_tools = set(scenario.expected_tools) | set(scenario.acceptable_tools)

        for criterion in scenario.success_criteria:
            criterion_lower = criterion.lower()
            passed = False

            # Check if an expected/acceptable tool was called
            if "should call" in criterion_lower or "should use" in criterion_lower:
                for tool_name in all_valid_tools:
                    if tool_name in searchable:
                        passed = True
                        break
                if not all_valid_tools and trace.tool_calls:
                    passed = True

            # Check for confirmation language
            elif "confirm" in criterion_lower:
                confirm_words = ["created", "deleted", "removed", "success", "done",
                                 "complete", "found", "exists", "returned"]
                passed = any(w in searchable for w in confirm_words)

            # Check for content presence — require quoted terms if present
            elif "contain" in criterion_lower or "include" in criterion_lower or "should" in criterion_lower:
                quoted = re.findall(r"'([^']+)'", criterion)
                if quoted:
                    passed = all(q.lower() in searchable for q in quoted)
                else:
                    # No quoted terms — check if there's meaningful tool response data
                    passed = bool(trace.tool_calls and any(tc.response for tc in trace.tool_calls))

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
        threshold = CRITICAL_THRESHOLDS.get("success_criteria", 0.7)
        return EvalScore(
            evaluator="success_criteria",
            score=score,
            passed=score >= threshold,
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


def _flatten_response(response: Any) -> str:
    """Recursively extract all string values from a tool response."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return " ".join(_flatten_response(v) for v in response.values())
    if isinstance(response, list):
        return " ".join(_flatten_response(item) for item in response)
    return str(response)


def compute_overall_pass(scores: list[EvalScore]) -> bool:
    """A scenario passes if all critical evaluators meet their thresholds."""
    for score in scores:
        threshold = CRITICAL_THRESHOLDS.get(score.evaluator)
        if threshold is not None and score.score < threshold:
            return False
    return True
