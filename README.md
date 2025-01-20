
# Boston Housing Predictions with Vertex AI

This repository shows how to deploy a simple machine learning model to Vertex AI for both online and batch predictions. The model is a feedforward neural network trained on the Boston Housing dataset.

---

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
├── pipeline_scripts
│   └── boston_pipeline.py           # Kubeflow pipeline definition
├── .gitignore                      
├── prediction_input.jsonl           # Example input file for batch predictions
├── README.md                        
├── requirements.txt                
```

---

## Prerequisites

1. **Google Cloud Platform (GCP)**:
   - A GCP project with billing enabled.
   - Enable the Vertex AI, Cloud Storage, Artifact Registry, and BigQuery APIs.

2. **Python Environment**:
   - Python 3.10+.
   - Install dependencies from `requirements.txt`.

3. **Docker**:
   - Installed and configured to build and push Docker images.

4. **Google Cloud SDK**:
   - Installed and authenticated with `gcloud auth application-default login`.

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

---

## Running the Pipeline

### Compile and Submit the Pipeline
Use the following command to compile and run the pipeline:
```bash
python run_boston_pipeline.py
```

### Monitor Pipeline Execution
1. Open the **Google Cloud Console**.
2. Navigate to **Vertex AI > Pipelines**.
3. Monitor the status of your pipeline execution.

---

## Testing

### Unit Tests
Run the tests:
```bash
pytest tests/
```

---

## Clean!

To avoid costs, delete resources after use:
```bash
gcloud ai models delete MODEL_ID --project=PROJECT_ID
gcloud ai endpoints delete ENDPOINT_ID --project=PROJECT_ID
```
