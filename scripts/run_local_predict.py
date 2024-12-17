import argparse
import json
import numpy as np
import os
import subprocess
import tensorflow as tf
from google.cloud import aiplatform


def main():
    parser = argparse.ArgumentParser(description="Run local predictions using a model from Vertex AI Model Registry.")
    parser.add_argument("--project", type=str, required=True, help="GCP project ID.")
    parser.add_argument("--location", type=str, required=True, help="GCP region for Vertex AI.")
    parser.add_argument("--model_id", type=str, required=True, help="Vertex AI Model resource ID.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSON file containing 'instances'.")
    parser.add_argument("--download_dir", type=str, default=".", help="Local directory to download model artifacts.")
    args = parser.parse_args()

    # Initialize Vertex AI
    aiplatform.init(project=args.project, location=args.location)

    # Retrieve the model from Model Registry
    model_resource_name = f"projects/{args.project}/locations/{args.location}/models/{args.model_id}"
    model = aiplatform.Model(model_resource_name)

    # The artifact_uri points to a GCS location where the model.keras file should be stored.
    artifact_uri = model.artifact_uri
    if not artifact_uri:
        raise ValueError("No artifact_uri found for this model. Ensure the model was uploaded correctly.")

    # We assume model.keras is stored at artifact_uri.
    model_keras_path_gcs = os.path.join(artifact_uri, "model.keras")

    # Create download directory if not exists
    os.makedirs(args.download_dir, exist_ok=True)
    local_model_path = os.path.join(args.download_dir, "model.keras")

    print(f"Downloading model from {model_keras_path_gcs} to {local_model_path}...")
    subprocess.check_call(["gsutil", "cp", model_keras_path_gcs, local_model_path])

    print(f"Loading model from {local_model_path}...")
    model = tf.keras.models.load_model(local_model_path)

    # Load the input data
    with open(args.input_file, "r") as f:
        data = json.load(f)

    if "instances" not in data:
        raise ValueError("JSON input file must contain an 'instances' key.")

    instances = np.array(data["instances"])

    print("Running predictions...")
    predictions = model.predict(instances).tolist()
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
