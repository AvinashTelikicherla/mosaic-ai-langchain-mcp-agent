"""Call the deployed endpoint.

    python query_endpoint.py <endpoint-name> "How do I stream tokens from an agent?"
"""

import sys

from dotenv import load_dotenv
from mlflow.deployments import get_deploy_client

load_dotenv()

endpoint, question = sys.argv[1], " ".join(sys.argv[2:])
response = get_deploy_client("databricks").predict(
    endpoint=endpoint, inputs={"input": [{"role": "user", "content": question}]}
)
for item in response["output"]:
    print(item)
