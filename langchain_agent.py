"""File 1: a simple ReAct agent (OpenAI + LangChain docs MCP server) with streaming.

Run it directly:
    python langchain_agent.py "How do I add memory to a LangChain agent?"
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

MCP_URL = "https://docs.langchain.com/mcp"
# submit_feedback writes to the docs team, so the agent only gets the read-only tools.
EXCLUDED_TOOLS = {"submit_feedback"}
SYSTEM_PROMPT = (
    "You are a LangChain documentation assistant. Always search the docs with the "
    "available tools before answering, prefer them over prior knowledge, and "
    "mention the doc pages you used."
)


async def build_agent():
    """Create a ReAct agent whose tools come from the LangChain docs MCP server."""
    client = MultiServerMCPClient(
        {"langchain_docs": {"transport": "streamable_http", "url": MCP_URL}}
    )
    tools = [t for t in await client.get_tools() if t.name not in EXCLUDED_TOOLS]
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    return create_agent(model, tools, system_prompt=SYSTEM_PROMPT)


async def stream_answer(agent, question: str) -> None:
    """Print tool usage and stream the answer token by token."""
    inputs = {"messages": [{"role": "user", "content": question}]}
    async for chunk, _metadata in agent.astream(inputs, stream_mode="messages"):
        if not isinstance(chunk, AIMessageChunk):
            continue  # skip ToolMessage results
        for call in chunk.tool_call_chunks:
            if call.get("name"):
                print(f"\n[tool call: {call['name']}]", file=sys.stderr)
        if isinstance(chunk.content, str):
            print(chunk.content, end="", flush=True)
    print()


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I stream tokens from a LangChain agent?"
    agent = await build_agent()
    await stream_answer(agent, question)


if __name__ == "__main__":
    asyncio.run(main())
