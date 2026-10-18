"""File 3: log the agent to Unity Catalog and deploy it as a serving endpoint.

    python deploy.py
"""

from __future__ import annotations

import argparse
import os

import mlflow
from databricks import agents
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument(
        "--log-only", action="store_true", help="register the model but do not deploy it"
    )
    args = parser.parse_args()

    uc_model = os.environ["UC_MODEL_NAME"]
    secret_scope = os.environ["DATABRICKS_SECRET_SCOPE"]
    secret_key = os.environ["DATABRICKS_SECRET_KEY"]

    mlflow.set_registry_uri("databricks-uc")

    # 1. Log (models-from-code) and register in Unity Catalog.
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model="mosaic_agent.py",
            code_paths=["langchain_agent.py"],  # imported by mosaic_agent.py
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                "langchain>=1.0",
                "langchain-openai>=1.0",
                "langchain-mcp-adapters>=0.1.9",
                "langgraph>=1.0",
                "python-dotenv",
                "pydantic>=2",
            ],
            registered_model_name=uc_model,
        )
    print(f"Registered {uc_model} version {info.registered_model_version}")

    # 2. Smoke test the logged artifact (needs OPENAI_API_KEY in your local env).
    if not args.skip_smoke_test:
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        reply = loaded.predict(
            {"input": [{"role": "user", "content": "How do I add memory to a LangChain agent?"}]}
        )
        print("Smoke test OK:", str(reply["output"][-1])[:200])

    if args.log_only:
        return  # deploy later with serving_endpoint.py

    # 3. Deploy. The OpenAI key is injected from a Databricks secret, never baked in.
    deployment = agents.deploy(
        uc_model,
        info.registered_model_version,
        environment_vars={
            "OPENAI_API_KEY": f"{{{{secrets/{secret_scope}/{secret_key}}}}}",
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        },
    )
    print("Endpoint:", deployment.endpoint_name)
    print("Query URL:", deployment.query_endpoint)


if __name__ == "__main__":
    main()
