# Use TensorFlow base image
FROM tensorflow/tensorflow:2.18.0

# Set working directory
WORKDIR /app

# Copy application files
COPY predict.py .
COPY requirements.txt .
COPY service-account.json ./service-account.json

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables required by Vertex AI
ENV AIP_MODEL_DIR=/app/model
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_HTTP_PORT=8080

# Expose the port for health and prediction routes
EXPOSE 8080

# Start the FastAPI app with Uvicorn
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]
