"""Data models for the evaluation framework."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Stage(str, Enum):
    SETUP = "setup"
    VERIFY = "verify"
    TEARDOWN = "teardown"


class ResultStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class ToolCallTrace(BaseModel):
    """Captures a single tool call and its response."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    response: Any = None
    error: str | None = None
    duration_ms: float | None = None


class ConversationTrace(BaseModel):
    """Full conversation trace for evaluation."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    final_response: str = ""
    total_duration_ms: float | None = None


class Scenario(BaseModel):
    """A test scenario loaded from YAML."""

    id: str
    name: str
    prompt: str
    expected_tools: list[str] = Field(default_factory=list)
    acceptable_tools: list[str] = Field(default_factory=list)
    expected_tool_params: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
    expected_response_contains: list[str] = Field(default_factory=list)
    stage: Stage
    depends_on: str | None = None


class EvalScore(BaseModel):
    """Score from an evaluator."""

    evaluator: str
    score: float
    passed: bool
    reasoning: str = ""


class ScenarioResult(BaseModel):
    """Result of running and evaluating a scenario."""

    scenario_id: str
    scenario_name: str
    stage: Stage
    status: ResultStatus = ResultStatus.FAILED
    trace: ConversationTrace | None = None
    scores: list[EvalScore] = Field(default_factory=list)
    passed: bool = False
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    state_captured: dict[str, Any] = Field(default_factory=dict)
