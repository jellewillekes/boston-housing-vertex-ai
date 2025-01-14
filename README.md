# Boston Housing Predictions with Vertex AI

This repository shows how to deploy a simple machine learning model to Vertex AI for both online and batch predictions. The model is a feedforward neural network trained on the Boston Housing dataset.

## Project Structure

```
├── config
│   └── config.yaml                  # Configuration file with GCP project details
├── local_model_dir
│   └── model.keras                  # Locally trained model
├── scripts
│   ├── create_prediction_input.py   
│   ├── json_payload.py              
│   ├── load_config.py              
│   ├── run_batch_prediction.py      # Runs a batch prediction job on Vertex AI
│   ├── run_custom_training_job.py   # Submits a custom training job to Vertex AI
│   ├── run_local_predict.py         # Runs predictions locally using a Vertex AI model
│   └── upload_model.py              
├── serving
│   ├── Dockerfile                
│   ├── predict.py                   # FastAPI application for online predictions
│   ├── requirements.txt           
│   └── run_example.py               # Example script to test the serving container locally
├── tests
│   ├── test_model.py               
│   └── test_predict.py              
├── training
│   ├── Dockerfile                   
│   ├── model.py                     # Define simple NN
│   ├── requirements.txt            
│   └── train.py                     # Training script for the model
├── .gitignore                      
├── prediction_input.jsonl           # Example input file for batch predictions
├── README.md                        
├── requirements.txt                
```

---

## Prerequisites

1. **Google Cloud Platform (GCP)**:
   - A GCP project with billing on.
   - Enable the Vertex AI and Cloud Storage APIs.

2. **Python Environment**:
   - Python 3.10+.
   - Install dependencies `requirements.txt`.

3. **Docker**:
   - Installed and set up to build and push Docker images.

4. **Google Cloud SDK**:
   - Installed and authenticated.

5. **Configure `config.yaml`**:
   - Update `config/config.yaml` with your project details:
     ```yaml
     project_id: "your-gcp-project-id"
     region: "your-region"
     bucket: "gs://your-bucket-name/"
     repo: "your-artifact-repository"
     ```

---

## Step-by-Step Guide

### 1. Build and Push Docker Images

#### Training Image
```bash
docker build -t region-docker.pkg.dev/project-id/repo/training-image:latest -f training/Dockerfile .
docker push region-docker.pkg.dev/project-id/repo/training-image:latest
```
Variables like `project-id`, `repo` and `region` have to be set in `config.yaml` such that they can be referred back to later.

#### Serving Image
```bash
docker build -t region-docker.pkg.dev/project-id/repo/serving-image:latest -f serving/Dockerfile .
docker push region-docker.pkg.dev/project-id/repo/serving-image:latest
```

### 2. Train the Model

#### Locally

```bash
python training/train.py --model_dir local_model_dir
```

#### On Vertex AI

```bash
python scripts/run_custom_training_job.py
```
This script submits a custom training job using the training Docker container.

### 3. Upload the Model

Upload the trained model to the Vertex AI Model Registry:

```bash
python scripts/upload_model.py
```

### 4. Deploy the Model

#### Online Prediction Endpoint
Use the Vertex AI Console or SDK to deploy the uploaded model to an endpoint for online predictions.

#### Batch Prediction Job
Run a batch prediction job using the uploaded model:

```bash
python scripts/run_batch_prediction.py
```

### 5. Test Predictions

#### Online Prediction (Locally)
Run the FastAPI server locally:

```bash
cd serving
uvicorn predict:app --host 0.0.0.0 --port 8080
```
Send a POST request:

```bash
curl -X POST http://127.0.0.1:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"instances": [[0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.0900, 1.0, 296.0, 15.3, 396.9, 4.98]]}'
```

#### Batch Prediction
Prepare a JSONL input file (e.g., `prediction_input.jsonl`) and upload it to Cloud Storage:

```jsonl
{"instances": [[0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.0900, 1.0, 296.0, 15.3, 396.9, 4.98]]}
{"instances": [[0.02731, 0.0, 7.07, 0.0, 0.469, 6.421, 78.9, 4.9671, 2.0, 242.0, 17.8, 396.9, 9.14]]}
```
Run the batch prediction job:

```bash
python scripts/run_batch_prediction.py
```

---

## Testing

### Unit Tests
Run the test folder:

```bash
pytest tests/
```

---

## Cleaning Up

To avoid unnecessary costs, delete resources after use. Keeping an endpoint running can be costly in GCloud!

```bash
gcloud ai models delete MODEL_ID --project=PROJECT_ID
gcloud ai endpoints delete ENDPOINT_ID --project=PROJECT_ID
```

---
