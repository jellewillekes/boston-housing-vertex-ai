from google.cloud import aiplatform, storage
import json
from load_config import PROJECT_ID, REGION, BUCKET, REPO

ENDPOINT_ID = "946164940073336832"
GCS_INPUT_FILE = "input/prediction_input.jsonl"  # Path to file in GCS


def load_jsonl_from_gcs(bucket_name: str, file_path: str) -> list:
    """
    Load and parse a JSON file (not JSONL) from Google Cloud Storage.
    """
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(REPO)
    blob = bucket.blob(file_path)

    # Read the file content
    file_content = blob.download_as_text()
    print(f"Loaded file '{file_path}' from bucket '{bucket_name}'.")

    # Parse the entire file as a single JSON object
    data = json.loads(file_content)

    # Extract instances
    instances = data.get("instances", [])
    return instances


def online_prediction(project: str, region: str, endpoint_id: str, instances: list):
    aiplatform.init(project=project, location=region)

    endpoint = aiplatform.Endpoint(project=PROJECT_ID, endpoint_name=f"projects/{project}/locations/{region}/endpoints"
                                                                     f"/{endpoint_id}")
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
