import os
import json
import logging
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Environment variables for model directory and mode
LOCAL_ENV = os.environ.get("LOCAL_ENV", "true").lower() == "true"
if LOCAL_ENV:
    root_dir = os.path.abspath(os.path.dirname(__file__))
    MODEL_DIR = os.path.join(root_dir, "../local_model_dir")
else:
    MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "/app/model")

logger.info(f"Environment: {'Local' if LOCAL_ENV else 'Cloud'}")
logger.info(f"Model Directory: {MODEL_DIR}")


# Load the model
def load_model(model_dir):
    try:
        model_path = os.path.join(model_dir, "model.keras")
        logger.info(f"Loading model from: {model_path}")
        loaded_model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")
        return loaded_model
    except Exception as e:
        logger.error(f"Failed to load the model: {str(e)}")
        return None


model = load_model(MODEL_DIR)


# Input schema
class PredictionInput(BaseModel):
    instances: list[list[float]]


# Input processing
def process_jsonl_input(data: str):
    instances = []
    try:
        for i, line in enumerate(data.splitlines(), start=1):
            logger.debug(f"Processing line {i}: {line.strip()}")
            obj = json.loads(line)
            if "instances" not in obj:
                raise ValueError(f"Missing 'instances' key in JSONL line {i}.")
            instances.append(obj["instances"])
        logger.info(f"Successfully processed {len(instances)} lines from JSONL file.")
    except Exception as e:
        logger.error(f"Error processing JSONL input: {str(e)}")
        raise ValueError("Invalid JSONL input.")
    return instances


def validate_and_transform_payload(data: dict):
    if not data:
        raise ValueError("No data received or JSON is invalid.")
    if "instances" not in data:
        raise ValueError("Missing 'instances' key in payload.")

    instances = data["instances"]

    if not isinstance(instances, list):
        raise ValueError("'instances' should be a list of inputs.")

    for instance in instances:
        if not isinstance(instance, list) or len(instance) != 13:
            raise ValueError("Each input instance must be a list of 13 numerical features.")

    try:
        instances_array = np.array(instances, dtype=np.float32)
        logger.info(f"Input converted to NumPy array with shape {instances_array.shape}.")
    except Exception as e:
        raise ValueError(f"Error converting instances to NumPy array: {e}")

    return instances_array


# Prediction function
def predict_instances(instances, model):
    if model is None:
        raise ValueError("Model is not loaded.")
    logger.info(f"Running predictions on input of shape {instances.shape}.")
    try:
        preds = model.predict(instances).tolist()
        logger.info(f"Predictions completed successfully. Output: {preds}")
        return preds
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise ValueError("Prediction error.")


# Routes
@app.get("/health")
def health():
    """Health check endpoint."""
    logger.info("Health route reached.")
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    """Prediction endpoint. Handles both JSON and JSONL input formats."""
    try:
        content_type = request.headers.get("content-type")
        if content_type == "application/jsonlines":
            logger.info("Processing JSONL input.")
            raw_body = await request.body()
            instances = process_jsonl_input(raw_body.decode("utf-8"))
        elif content_type == "application/json":
            logger.info("Processing standard JSON input.")
            body = await request.json()
            instances = validate_and_transform_payload(body)
        else:
            raise HTTPException(status_code=415, detail="Unsupported Content-Type.")

        # Convert to NumPy array
        instances_array = np.array(instances, dtype=np.float32)

        # Make predictions
        predictions = predict_instances(instances_array, model)

        return {"predictions": predictions}

    except ValueError as e:
        logger.error(f"Validation or prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed.")
