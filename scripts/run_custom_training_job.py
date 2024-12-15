# scripts/run_custom_training_job.py
from google.cloud import aiplatform
from load_config import PROJECT_ID, REGION, BUCKET, REPO

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}")

training_image_uri = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/training-image:latest"

job = aiplatform.CustomContainerTrainingJob(
    display_name="boston-housing",
    container_uri=training_image_uri,
)

job.run(
    args=["--model_dir", f"{BUCKET}artifacts/"],
    replica_count=1,
    machine_type="n1-standard-4",
    base_output_dir=f"{BUCKET}artifacts/"
)

