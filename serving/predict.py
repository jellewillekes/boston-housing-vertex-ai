import os
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
from google.cloud import logging
from google.oauth2 import service_account
from google.logging.type import log_severity_pb2 as severity
import tensorflow as tf
import requests

# Configuration
PROJECT_ID = "affor-models"
MODEL_BUCKET = "boston-example"
MODEL_FILENAME = "artifacts/model.keras"
local_model_path = "model.keras"


def get_service_account_from_metadata():
    METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email"
    HEADERS = {"Metadata-Flavor": "Google"}
    try:
        response = requests.get(METADATA_URL, headers=HEADERS)
        if response.status_code == 200:
            return response.text.strip()
    except Exception as e:
        print(f"Error retrieving service account from metadata: {e}")
    return None


metadata_sa_email = get_service_account_from_metadata()
print(
    f"RUNNING under service account (from metadata): {metadata_sa_email if metadata_sa_email else 'Could not retrieve from metadata'}")

# Cloud Logging Setup
credentials = service_account.Credentials.from_service_account_file(
    './service-account.json')
client = logging.Client(credentials=credentials)
logger = client.logger('log_name')

# FastAPI App
app = FastAPI()

HEALTH_ROUTE = os.getenv("AIP_HEALTH_ROUTE", "/health")
PREDICTIONS_ROUTE = os.getenv("AIP_PREDICT_ROUTE", "/predict")

model = None


def load_model():
    global model
    try:
        if model is None:
            gcs_client = storage.Client(project=PROJECT_ID)
            bucket = gcs_client.bucket(MODEL_BUCKET)
            blob = bucket.blob(MODEL_FILENAME)

            # Download model to local path
            blob.download_to_filename(local_model_path)
            print(f"Model downloaded to '{local_model_path}'.")

            # Load the model
            model = tf.keras.models.load_model(local_model_path)
            print("Model loaded successfully.")
        else:
            print("Model is already loaded, skip reload.")

    except Exception as e:
        logger.log_struct({"error": str(e)}, severity=severity.ERROR)
        raise RuntimeError(f"Failed to load model: {e}")


# Load the model once when the app starts
try:
    load_model()
except Exception as e:
    print(f"Critical error during startup: {e}")
    exit(1)  # Exit the app if the model fails to load


@app.get(HEALTH_ROUTE, status_code=200)
def health():
    logger.log_text("Health check OK")
    return {"message": "Healthy Server!"}


@app.post(PREDICTIONS_ROUTE)
async def predict(request: Request):
    try:
        logger.log_text("Prediction request received")

        # Extract and validate JSON input
        body = await request.json()
        instances = body.get("instances", [])
        if not instances or not isinstance(instances, list):
            raise ValueError("Invalid JSON input: 'instances' must be a non-empty list.")

        # Convert input to TensorFlow tensor
        inputs = tf.convert_to_tensor(instances, dtype=tf.float32)
        logger.log_struct({"input_shape": inputs.shape}, severity=severity.INFO)

        # Perform predictions
        predictions = model.predict(inputs).tolist()
        logger.log_struct({"predictions": predictions}, severity=severity.INFO)

        return {"predictions": predictions}

    except Exception as error:
        error_response = {"error": str(error)}
        logger.log_struct(error_response, severity=severity.ERROR)
        raise HTTPException(status_code=400, detail=error_response)
