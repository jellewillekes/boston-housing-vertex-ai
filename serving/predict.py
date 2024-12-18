import os
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage, logging
from google.oauth2 import service_account
from google.logging.type import log_severity_pb2 as severity

app = FastAPI()

HEALTH_ROUTE = os.environ["AIP_HEALTH_ROUTE"]
PREDICTIONS_ROUTE = os.environ["AIP_PREDICT_ROUTE"]

# Cloud Logging setup
client = logging.Client()
logger = client.logger("vertex-ai-batch-predictions")


@app.get(HEALTH_ROUTE, status_code=200)
def health():
    logger.log_text("Health check OK", severity=severity.INFO)
    return {"Server Status": "OK"}


@app.post(PREDICTIONS_ROUTE)
async def predict(request: Request):
    try:
        body = await request.json()
        instances = body.get("instances", [])

        # Placeholder predictions
        predictions = [0.1 for _ in instances]
        logger.log_text(f"Successful predictions: {predictions}", severity=severity.INFO)

        return {"predictions": predictions}

    except Exception as error:
        logger.log_struct({"error": str(error)}, severity=severity.ERROR)
        raise HTTPException(status_code=400, detail={"error": str(error)})
