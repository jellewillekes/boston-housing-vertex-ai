from google.cloud import aiplatform
from load_config import PROJECT_ID, REGION

MODEL_ID = "3082771119539748864"
ENDPOINT_NAME = "boston-housing-code"
DEPLOYED_MODEL_NAME = "boston-housing-deployed"


def deploy_model_to_vertex_ai(project_id: str, region: str, model_id: str, endpoint_name: str,
                              deployed_model_name: str):
    """
    Deploy a model from the Vertex AI Model Registry to an endpoint.
    """
    # Initialize
    aiplatform.init(project=project_id, location=region)

    # Get registered model
    model = aiplatform.Model(model_name=model_id)
    print(f"Retrieved model: {model.resource_name}")

    # Create or get the endpoint
    endpoint = None
    endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
    if endpoints:
        endpoint = endpoints[0]
        print(f"Using existing endpoint: {endpoint.display_name}")
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
        print(f"Created new endpoint: {endpoint.display_name}")

    # Deploy the model to the endpoint
    print("Deploying model to endpoint...")
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_name,
        machine_type="n1-standard-4",
        service_account='712583227660-compute@developer.gserviceaccount.com'
    )

    print(f"Model deployed to endpoint: {endpoint.resource_name}")


if __name__ == "__main__":
    deploy_model_to_vertex_ai(PROJECT_ID, REGION, MODEL_ID, ENDPOINT_NAME, DEPLOYED_MODEL_NAME)
