"""Alternative deployment: create the serving endpoint directly with the Databricks SDK.

Use this when you want full control over the endpoint (name, size, scaling) instead of
the agent-specific `agents.deploy` used in deploy.py. Run `deploy.py --log-only` first
(or log the model yourself) so a Unity Catalog model version exists.

    python serving_endpoint.py --version 1
    python serving_endpoint.py --version 2          # updates the existing endpoint
"""

from __future__ import annotations

import argparse
import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Unity Catalog model version")
    parser.add_argument("--name", default="langchain-docs-agent-endpoint")
    parser.add_argument("--workload-size", default="Small", choices=["Small", "Medium", "Large"])
    parser.add_argument("--no-scale-to-zero", action="store_true")
    args = parser.parse_args()

    scope = os.environ["DATABRICKS_SECRET_SCOPE"]
    key = os.environ["DATABRICKS_SECRET_KEY"]

    config = EndpointCoreConfigInput(
        served_models=[
            ServedModelInput(
                model_name=os.environ["UC_MODEL_NAME"],
                model_version=args.version,
                workload_size=args.workload_size,
                scale_to_zero_enabled=not args.no_scale_to_zero,
                # The OpenAI key is resolved from a Databricks secret at runtime.
                environment_vars={
                    "OPENAI_API_KEY": f"{{{{secrets/{scope}/{key}}}}}",
                    "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                },
            )
        ]
    )

    w = WorkspaceClient()
    try:
        w.serving_endpoints.get(args.name)
        print(f"Endpoint '{args.name}' exists, updating to version {args.version}...")
        waiter = w.serving_endpoints.update_config(
            name=args.name, served_models=config.served_models
        )
    except NotFound:
        print(f"Creating endpoint '{args.name}'...")
        waiter = w.serving_endpoints.create(name=args.name, config=config)

    endpoint = waiter.result()  # blocks until the endpoint is READY (can take several minutes)
    print(f"Endpoint ready: {endpoint.name} ({endpoint.state.ready})")


if __name__ == "__main__":
    main()
