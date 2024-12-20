import os
import json
import numpy as np
from google.cloud import storage
import tensorflow as tf

# Configuration
PROJECT_ID = "affor-models"
MODEL_BUCKET = "boston-example"
MODEL_FILENAME = "artifacts/model.keras"
GCS_INPUT_FILE = "input/prediction_input.jsonl"
local_model_path = "model.keras"


def load_model_from_gcs(bucket_name: str, model_path: str, local_path: str) -> tf.keras.Model:
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_path)

        blob.download_to_filename(local_path)
        print(f"Model downloaded from GCS bucket '{bucket_name}' to '{local_path}'.")

        model = tf.keras.models.load_model(local_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from GCS: {e}")


def load_jsonl_from_gcs(bucket_name: str, file_path: str) -> np.ndarray:
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)

        file_content = blob.download_as_text()
        print(f"Loaded file '{file_path}' from bucket '{bucket_name}'.")

        instances = []
        for line in file_content.strip().split("\n"):
            instance = json.loads(line)
            if "instances" in instance:
                instances.extend(instance["instances"])
            else:
                print(f"Warning: Line missing 'instances' key: {line}")

        instances_array = np.asarray(instances, dtype=np.float32)
        print(f"Parsed instances with shape: {instances_array.shape}")
        return instances_array
    except Exception as e:
        raise RuntimeError(f"Failed to load JSONL file from GCS: {e}")


def main():
    model = load_model_from_gcs(bucket_name=MODEL_BUCKET, model_path=MODEL_FILENAME, local_path=local_model_path)

    prediction_input = load_jsonl_from_gcs(bucket_name=MODEL_BUCKET, file_path=GCS_INPUT_FILE)

    if len(prediction_input.shape) == 1:
        prediction_input = prediction_input.reshape(1, -1)

    predictions = model.predict(prediction_input).tolist()
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
