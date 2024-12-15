# scripts/upload_model.py
from google.cloud import aiplatform
from load_config import PROJECT_ID, REGION, BUCKET, REPO

aiplatform.init(project=PROJECT_ID, location=REGION)

# Serving container image URI for custom predictions
serving_image_uri = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/serving-image:latest"

# Point to the specific directory where the trained model is stored
artifact_uri = f"{BUCKET}artifacts/"

model = aiplatform.Model.upload(
    display_name="boston-housing-model",
    artifact_uri=artifact_uri,
    serving_container_image_uri=serving_image_uri,
    serving_container_predict_route="/predict",
    serving_container_health_route="/health"
)

print("Model uploaded:", model.resource_name)
