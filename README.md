# LangChain ReAct agent on Mosaic AI

A simple ReAct agent (OpenAI + the [LangChain docs MCP server](https://docs.langchain.com/mcp))
wrapped as an MLflow `ResponsesAgent` and deployed as a Databricks Mosaic AI serving endpoint.

| File | What it does |
|---|---|
| `langchain_agent.py` | The agent: `create_agent` + MCP tools + token streaming. Runs standalone. |
| `mosaic_agent.py` | `LangChainMCPAgent(ResponsesAgent)`: the Mosaic AI wrapper, logged as code. |
| `deploy.py` | Logs to Unity Catalog, smoke-tests, deploys the serving endpoint. |
| `serving_endpoint.py` | Alternative: create/update the endpoint with the Databricks SDK. |
| `query_endpoint.py` | Calls the deployed endpoint. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # fill in OPENAI_API_KEY and Databricks values
```

## 1. Run the agent locally (streaming)

```bash
python langchain_agent.py "How do I add memory to a LangChain agent?"
```

## 2. Deploy as a serving endpoint

Store the OpenAI key as a Databricks secret (the endpoint reads it at runtime):

```bash
databricks secrets create-scope openai
databricks secrets put-secret openai api_key
```

Then:

```bash
python deploy.py
```

This logs `mosaic_agent.py` to Unity Catalog, runs a smoke test, and calls `agents.deploy`
with the secret injected as `OPENAI_API_KEY`. Use `--skip-smoke-test` to skip the local check.

### Alternative: plain SDK endpoint

```bash
python deploy.py --log-only --skip-smoke-test     # register a UC model version
python serving_endpoint.py --version 1            # create (or update) the endpoint
```

| | `agents.deploy` (`deploy.py`) | `serving_endpoints.create` (`serving_endpoint.py`) |
|---|---|---|
| Level | High-level, agent-specific helper | Generic Databricks SDK call |
| Endpoint name | Generated for you | You choose it |
| Agent extras | Designed for agents (e.g. review app, feedback and tracing wiring) | None; you get a plain endpoint |
| Tuning | A few options | Full control (size, scaling, traffic, rate limits, tags) |
| Best for | Fastest path for agents | Custom or CI/CD-managed endpoints |

## 3. Query it

```bash
python query_endpoint.py <endpoint-name> "How do I stream tokens from an agent?"
```

## Notes

- MCP tools are async-only while `ResponsesAgent.predict_stream` is sync, so the wrapper drives
  the async agent stream on a private event loop and yields each message as it arrives.
- The agent is built lazily on the first request, so logging the model needs no network access.
- `submit_feedback` is excluded; the agent only gets the read-only docs tools.
- The endpoint needs outbound internet access to reach `api.openai.com` and `docs.langchain.com`.
