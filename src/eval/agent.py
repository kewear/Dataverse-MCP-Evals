"""LLM Agent with MCP tool calling via GitHub Models API."""

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

    Uses the OpenAI SDK pointed at GitHub Models API for inference,
    and routes tool calls through the MCP client.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        github_token: str,
        model: str = "gpt-4o",
        base_url: str = "https://models.inference.ai.azure.com",
        max_tool_rounds: int = 10,
    ):
        self.mcp_client = mcp_client
        self.model = model
        self.max_tool_rounds = max_tool_rounds

        self._openai = OpenAI(
            api_key=github_token,
            base_url=base_url,
        )

    def run(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> ConversationTrace:
        """Run the agent with a prompt and return the full conversation trace.

        The agent will:
        1. Send the prompt to the LLM with available MCP tools
        2. If the LLM requests tool calls, execute them via MCP
        3. Feed tool results back to the LLM
        4. Repeat until the LLM produces a final text response or max rounds reached
        """
        start = time.perf_counter()
        trace = ConversationTrace()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        trace.messages.append({"role": "user", "content": prompt})

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

        trace.total_duration_ms = (time.perf_counter() - start) * 1000
        return trace
