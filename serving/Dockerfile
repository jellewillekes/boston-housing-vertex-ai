FROM tensorflow/tensorflow:2.18.0

WORKDIR /app

COPY predict.py .
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --ignore-installed -r requirements.txt

ENV AIP_MODEL_DIR=/app/model
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HEALTH_ROUTE=/health
ENV LOCAL_ENV=false

# Expose the default port
EXPOSE 8080

# Start the FastAPI app with Uvicorn
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]
