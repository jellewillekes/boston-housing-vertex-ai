import os
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
import tensorflow as tf
import numpy as np
from contextlib import asynccontextmanager

# Model storage details
MODEL_BUCKET = "boston-example"
MODEL_FILENAME = "artifacts/model.keras"
local_model_path = "/tmp/model.keras"


# Initialize GCS client
client = storage.Client(project='affor-models')

# Download model from GCS bucket
bucket = client.bucket(MODEL_BUCKET)
blob = bucket.blob(MODEL_FILENAME)
blob.download_to_filename(local_model_path)

# Load and return the model
loaded_model = tf.keras.models.load_model(local_model_path)

