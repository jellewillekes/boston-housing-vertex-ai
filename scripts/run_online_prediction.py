from google.cloud import aiplatform, storage
import json
from load_config import PROJECT_ID, REGION, BUCKET, REPO

ENDPOINT_ID = "4219437444241555456"
GCS_INPUT_FILE = "input/prediction_input.jsonl"  # Path to file in GCS


def load_jsonl_from_gcs(bucket_name: str, file_path: str) -> list:
    """
    Load and parse a JSONL file from Google Cloud Storage.

    Args:
        bucket_name (str): Name of the GCS bucket.
        file_path (str): Path to the JSONL file in the bucket.

    Returns:
        list: A list of instances parsed from the JSONL file.
    """
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(REPO)
    blob = bucket.blob(file_path)

    # Read the file content
    file_content = blob.download_as_text()
    print(f"Loaded file '{file_path}' from bucket '{bucket_name}'.")

    # Parse JSONL file line by line
    instances = []
    for line in file_content.strip().split("\n"):
        instance = json.loads(line)
        if "instances" in instance:
            instances.extend(instance["instances"])  # Add instances to the list
        else:
            print(f"Warning: Line missing 'instances' key: {line}")

    return instances


def online_prediction(project: str, region: str, endpoint_id: str, instances: list):
    """
    Perform an online prediction using a deployed model on Vertex AI.

    Args:
        project (str): Google Cloud project ID.
        region (str): Google Cloud region.
        endpoint_id (str): Vertex AI endpoint ID.
        instances (list): Input data instances for prediction.
    """
    aiplatform.init(project=project, location=region)

    endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{project}/locations/{region}/endpoints/{endpoint_id}")
    print(f"Using endpoint: {endpoint.resource_name}")

    try:
        response = endpoint.predict(instances=instances)
        print("Prediction response:", response.predictions)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    # Load instances from GCS file
    print("Loading input data from GCS...")
    instances = load_jsonl_from_gcs(bucket_name=BUCKET, file_path=GCS_INPUT_FILE)

    if not instances:
        print("Error: No instances found in the input file.")
    else:
        # Perform online prediction
        print("Sending online prediction request...")
        online_prediction(PROJECT_ID, REGION, ENDPOINT_ID, instances)
