"""LLM Agent with MCP tool calling via Azure AI Foundry."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI

from .mcp_client import MCPClient
from .models import ConversationTrace, ToolCallTrace

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful assistant that interacts with Microsoft Dataverse via MCP tools.
When the user asks you to perform an action on Dataverse (create tables, records, \
search, manage skills, etc.), use the available tools to fulfill the request.
Always confirm what you did after completing the action.
If a tool call fails, report the error clearly.
"""


class Agent:
    """LLM agent that uses MCP tools to interact with Dataverse.

    Uses the OpenAI SDK with Azure AI Foundry v1 endpoint for inference,
    and routes tool calls through the MCP client.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        azure_endpoint: str,
        api_key: str,
        deployment: str = "gpt-4.1",
        max_tool_rounds: int = 10,
    ):
        self.mcp_client = mcp_client
        self.model = deployment
        self.max_tool_rounds = max_tool_rounds

        base = azure_endpoint.rstrip("/")
        # If the endpoint already includes /openai/v1, use it as-is
        if base.endswith("/openai/v1"):
            base_url = base
        else:
            base_url = f"{base}/openai/v1"

        self._openai = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def run(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> ConversationTrace:
        """Run the agent with a single prompt and return the conversation trace."""
        start = time.perf_counter()
        trace = ConversationTrace()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        trace.messages.append({"role": "user", "content": prompt})

        self._run_loop(messages, trace)
        trace.total_duration_ms = (time.perf_counter() - start) * 1000
        return trace

    def run_conversation(
        self,
        prompts: list[str],
        system_prompt: str = SYSTEM_PROMPT,
        _existing_messages: list[dict[str, Any]] | None = None,
    ) -> list[ConversationTrace]:
        """Run multiple prompts in a single conversation (shared context).

        Each prompt builds on the previous conversation state, so the agent
        can reference prior tool outputs naturally. Returns a trace per step.

        If _existing_messages is provided, continues from that conversation state
        (and mutates it in place so the caller keeps the updated state).
        """
        if _existing_messages is not None:
            messages = _existing_messages
        else:
            messages = [
                {"role": "system", "content": system_prompt},
            ]
        tools = self.mcp_client.get_openai_tools()
        traces: list[ConversationTrace] = []

        for prompt in prompts:
            start = time.perf_counter()
            trace = ConversationTrace()

            messages.append({"role": "user", "content": prompt})
            trace.messages.append({"role": "user", "content": prompt})

            self._run_loop(messages, trace, tools=tools)
            trace.total_duration_ms = (time.perf_counter() - start) * 1000
            traces.append(trace)

        return traces

    def _run_loop(
        self,
        messages: list[dict[str, Any]],
        trace: ConversationTrace,
        tools: list[dict] | None = None,
    ) -> None:
        """Core agent loop — call LLM, execute tools, repeat until final response."""
        if tools is None:
            tools = self.mcp_client.get_openai_tools()

        for round_num in range(self.max_tool_rounds):
            logger.info("Agent round %d", round_num + 1)

            response = self._openai.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
            )

            choice = response.choices[0]
            message = choice.message

            # If no tool calls, we have the final response
            if not message.tool_calls:
                final_text = message.content or ""
                trace.final_response = final_text
                trace.messages.append({"role": "assistant", "content": final_text})
                messages.append({"role": "assistant", "content": final_text})
                break

            # Process tool calls
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            messages.append(assistant_msg)
            trace.messages.append(assistant_msg)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                logger.info("Calling tool: %s(%s)", tool_name, json.dumps(arguments)[:100])

                result = self.mcp_client.call_tool(tool_name, arguments)

                tool_trace = ToolCallTrace(
                    tool_name=tool_name,
                    arguments=arguments,
                    response=result["result"],
                    error=result["error"],
                    duration_ms=result["duration_ms"],
                )
                trace.tool_calls.append(tool_trace)

                # Format tool response for the LLM
                tool_content = json.dumps(result["result"]) if result["result"] else result["error"] or "No response"
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content,
                }
                messages.append(tool_msg)
                trace.messages.append(tool_msg)
        else:
            # Max rounds reached
            trace.final_response = "(Max tool rounds reached without final response)"
            logger.warning("Agent reached max tool rounds (%d)", self.max_tool_rounds)
