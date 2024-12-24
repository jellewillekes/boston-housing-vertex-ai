FROM python:3.10-slim

WORKDIR /app

COPY models/deployment/requirements-keyrings.txt ./requirements-keyrings.txt
COPY models/deployment/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-keyrings.txt \
    && pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY models/deployment/deploy.py ./deploy.py
COPY models/deployment/serve.py ./serve.py

# Default command: serve the model
ENV PORT 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "serve:app"]
