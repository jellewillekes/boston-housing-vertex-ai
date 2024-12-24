import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.algorithms.ada_boost_q import AdaBoostQ
from flask import Flask, request, jsonify
from google.cloud import storage
import pandas as pd
import os

# Configuration
PROJECT_ID = "affor-models"
BUCKET_NAME = "test-baseline"
MODEL_PATH = "models/trained/model/adaboostq_trained.json"

app = Flask(__name__)


def load_model_from_gcs():
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH)
    with blob.open("r") as f:
        model = AdaBoostQ.load(f)
    return model


model = load_model_from_gcs()


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return "OK", 200


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint."""
    input_data = request.get_json()
    if not input_data or "features" not in input_data:
        return jsonify({"error": "No features provided"}), 400

    rows = input_data["features"]
    if not rows:
        return jsonify({"error": "Empty feature list"}), 400

    # Example parsing: expects a list of dicts with keys ["date", "sec_id", ... features ...]
    df = pd.DataFrame(rows)

    # Convert 'date' column to datetime and set a MultiIndex if your model expects that
    if "date" in df.columns and "sec_id" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index(["date", "sec_id"], inplace=True)

    # Predict
    predictions = model.predict(df)

    return jsonify({"predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
