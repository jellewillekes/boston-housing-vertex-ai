from google.cloud import aiplatform
from load_config import PROJECT_ID, REGION


def undeploy_models(endpoint_id: str):
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{endpoint_id}")

    for deployed_model in endpoint.gca_resource.deployed_models:
        endpoint.undeploy(deployed_model_id=deployed_model.id)
        print(f"Undeployed model ID: {deployed_model.id}")


if __name__ == "__main__":
    ENDPOINT_ID = "4219437444241555456"
    undeploy_models(ENDPOINT_ID)
