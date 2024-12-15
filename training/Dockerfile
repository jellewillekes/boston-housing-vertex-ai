# training/Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY train.py .
COPY model.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "train.py"]
