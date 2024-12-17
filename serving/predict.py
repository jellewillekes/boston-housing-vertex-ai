import os
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage

app = FastAPI()

HEALTH_ROUTE = os.environ["AIP_HEALTH_ROUTE"]
PREDICTIONS_ROUTE = os.environ["AIP_PREDICT_ROUTE"]


@app.get(HEALTH_ROUTE, status_code=200)
def health():
    return {"Healthy Server!"}


@app.post(PREDICTIONS_ROUTE)
async def predict(request: Request):

    # response
    return {"predictions": [0.1, 0.9]}
