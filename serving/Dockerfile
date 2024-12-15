# serving/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY predict.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV AIP_MODEL_DIR /app/model
ENV LOCAL_ENV=false

EXPOSE 8080

ENTRYPOINT ["python", "predict.py"]
