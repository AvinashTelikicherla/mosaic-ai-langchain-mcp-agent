"""File 2: wrap the LangChain agent for the Mosaic AI Agent Framework.

MLflow's `ResponsesAgent` is the interface Databricks Model Serving expects. This file
is the "models-from-code" entry point that `deploy.py` logs.

Events are built explicitly with ResponsesAgentStreamEvent + the ResponsesAgent helper
methods (create_text_delta, create_function_call_item, ...), so we control exactly what
the client sees, including token-level text deltas.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import Generator
from typing import Any

import mlflow
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from langchain_agent import build_agent

mlflow.langchain.autolog()  # every LLM call and MCP tool call becomes an MLflow trace

_DONE = "response.output_item.done"


def _text(content: Any) -> str:
    """LangChain content is a str, or a list of blocks (MCP tools return blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "") for b in content if b
        )
    return ""


class LangChainMCPAgent(ResponsesAgent):
    """Serves the ReAct agent behind the OpenAI Responses-compatible interface."""

    def __init__(self, recursion_limit: int = 25) -> None:
        self._recursion_limit = recursion_limit
        self._agent = None  # built lazily so importing/logging needs no network access
        self._lock = threading.Lock()

    def _get_agent(self, loop: asyncio.AbstractEventLoop):
        with self._lock:
            if self._agent is None:
                self._agent = loop.run_until_complete(build_agent())
            return self._agent

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # Non-streaming = run the stream and keep only the completed items.
        outputs = [e.item for e in self.predict_stream(request) if e.type == _DONE]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        # MCP tools are async-only but ResponsesAgent is sync: drive the async stream
        # step by step on a private loop so events flow out as they are produced.
        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        loop = asyncio.new_event_loop()
        stream = None
        text_item_id: str | None = None  # id shared by deltas and their final "done" item
        try:
            agent = self._get_agent(loop)
            stream = agent.astream(
                {"messages": messages},
                {"recursion_limit": self._recursion_limit},
                # "messages" -> LLM tokens as they arrive; "updates" -> completed messages
                stream_mode=["messages", "updates"],
            )
            while True:
                try:
                    mode, data = loop.run_until_complete(stream.__anext__())
                except StopAsyncIteration:
                    break

                if mode == "messages":
                    chunk, _metadata = data
                    delta = _text(chunk.content) if isinstance(chunk, AIMessageChunk) else ""
                    if delta:
                        text_item_id = text_item_id or chunk.id or f"msg_{uuid.uuid4().hex}"
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=delta, item_id=text_item_id)
                        )

                elif mode == "updates":
                    for node_data in data.values():
                        if not isinstance(node_data, dict):
                            continue
                        for msg in node_data.get("messages", []):
                            yield from self._completed_items(msg, text_item_id)
                            if isinstance(msg, AIMessage) and _text(msg.content):
                                text_item_id = None  # next answer gets a fresh item id
        finally:
            if stream is not None:
                loop.run_until_complete(stream.aclose())
            loop.close()

    def _completed_items(
        self, msg: Any, text_item_id: str | None
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Turn one finished LangChain message into `response.output_item.done` events."""
        if isinstance(msg, AIMessage):
            text = _text(msg.content)
            if text:
                yield ResponsesAgentStreamEvent(
                    type=_DONE,
                    item=self.create_text_output_item(
                        text=text, id=text_item_id or msg.id or f"msg_{uuid.uuid4().hex}"
                    ),
                )
            for call in msg.tool_calls:
                yield ResponsesAgentStreamEvent(
                    type=_DONE,
                    item=self.create_function_call_item(
                        id=call["id"],
                        call_id=call["id"],
                        name=call["name"],
                        arguments=json.dumps(call["args"]),
                    ),
                )
        elif isinstance(msg, ToolMessage):
            yield ResponsesAgentStreamEvent(
                type=_DONE,
                item=self.create_function_call_output_item(
                    call_id=msg.tool_call_id, output=_text(msg.content)
                ),
            )


set_model(LangChainMCPAgent())