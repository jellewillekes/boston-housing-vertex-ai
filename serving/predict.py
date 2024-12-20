import os
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
from google.cloud import logging
from google.oauth2 import service_account
from google.logging.type import log_severity_pb2 as severity
import tensorflow as tf
import numpy as np

AIP_STORAGE_URI = os.environ["AIP_STORAGE_URI"]
HEALTH_ROUTE = os.environ["AIP_HEALTH_ROUTE"]
PREDICT_ROUTE = os.environ["AIP_PREDICT_ROUTE"]

# Logging with a separate service account
credentials = service_account.Credentials.from_service_account_file('./service-account.json')
logging_client = logging.Client(credentials=credentials)
logger = logging_client.logger('log_name')

# FastAPI App
app = FastAPI()
# Download and load model
model = None


def load_model():
    global model
    if model is not None:
        print("Model is already loaded, skipping reload.")
        return

    try:
        print("Loading model from AIP_STORAGE_URI...")
        gcs_client = storage.Client()  # ADC used here, no explicit credentials

        # AIP_STORAGE_URI:
        #  'model.keras' is stored at AIP_STORAGE_URI/model.keras
        model_path = f"{AIP_STORAGE_URI}/model.keras"

        # Parse bucket and object name from AIP_STORAGE_URI
        if not model_path.startswith("gs://"):
            raise RuntimeError("AIP_STORAGE_URI does not start with gs://")

        uri_no_scheme = model_path[len("gs://"):]
        parts = uri_no_scheme.split("/", 1)
        bucket_name = parts[0]
        object_path = parts[1] if len(parts) > 1 else "model.keras"

        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(object_path)

        local_model_path = "model.keras"
        blob.download_to_filename(local_model_path)
        print(f"Model downloaded to '{local_model_path}'.")

        model = tf.keras.models.load_model(local_model_path)
        print("Model loaded successfully.")

    except Exception as e:
        logger.log_struct({"error": str(e)}, severity=severity.ERROR)
        raise RuntimeError(f"Failed to load model: {e}")


# Load model at startup
try:
    load_model()
except Exception as e:
    print(f"Critical error during startup: {e}")
    exit(1)  # If the model fails to load at startup, exit.


@app.get(HEALTH_ROUTE, status_code=200)
def health():
    logger.log_text("Health check OK")
    return {"message": "Healthy Server!"}


@app.post(PREDICT_ROUTE)
async def predict(request: Request):
    try:
        logger.log_text("Prediction request received")
        body = await request.json()

        instances = body.get("instances", [])
        if not instances or not isinstance(instances, list):
            raise ValueError("Invalid JSON input: 'instances' must be a non-empty list.")

        # Convert to TF tensor
        inputs = tf.convert_to_tensor(instances, dtype=tf.float32)
        logger.log_struct({"input_shape": str(inputs.shape)}, severity=severity.INFO)

        # Make predictions
        predictions = model.predict(inputs).tolist()
        logger.log_struct({"predictions": predictions}, severity=severity.INFO)

        return {"predictions": predictions}

    except ValueError as ve:
        error_message = str(ve)
        logger.log_struct({"error": error_message}, severity=severity.ERROR)
        raise HTTPException(status_code=400, detail=error_message)

    except Exception as error:
        error_response = {"error": str(error)}
        logger.log_struct(error_response, severity=severity.ERROR)
        raise HTTPException(status_code=500, detail="Internal Server Error")
