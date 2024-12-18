import os
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
from google.cloud import logging
from google.oauth2 import service_account
from google.logging.type import log_severity_pb2 as severity
import tensorflow as tf

# Configuration
PROJECT_ID = "affor-models"
MODEL_BUCKET = "boston-example"
MODEL_FILENAME = "artifacts/model.keras"
local_model_path = "model.keras"

# Cloud Logging Setup
credentials = service_account.Credentials.from_service_account_file(
    './service-account.json')  # Ensure the correct path
client = logging.Client(credentials=credentials)
logger = client.logger('log_name')

# FastAPI App
app = FastAPI()

HEALTH_ROUTE = os.getenv("AIP_HEALTH_ROUTE", "/health")
PREDICTIONS_ROUTE = os.getenv("AIP_PREDICT_ROUTE", "/predict")

# Global variable to store the model
model = None


# Function to load the model from Google Cloud Storage
def load_model():
    global model
    try:
        if model is None:
            # Initialize GCS client
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
            print("Model is already loaded, skipping reload.")

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

        # Extract input from request
        body = await request.json()
        instances = body.get("instances", [])
        if not instances:
            raise ValueError("No instances provided in the request")

        inputs = tf.convert_to_tensor(instances, dtype=tf.float32)

        # Perform predictions
        predictions = model.predict(inputs).tolist()
        logger.log_struct({"predictions": predictions}, severity=severity.INFO)

        return {"predictions": predictions}

    except Exception as error:
        error_response = {"error": str(error)}
        logger.log_struct(error_response, severity=severity.ERROR)
        raise HTTPException(status_code=400, detail=error_response)
