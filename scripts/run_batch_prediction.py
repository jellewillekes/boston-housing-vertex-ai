from google.cloud import aiplatform
from load_config import PROJECT_ID, REGION, BUCKET

aiplatform.init(project=PROJECT_ID, location=REGION)


MODEL_ID = "1631486139619606528"  # Replace after upload step
INPUT_URI = f"{BUCKET}input/prediction_input.jsonl"
OUTPUT_PREFIX = f"{BUCKET}output/"

model_resource_name = f"projects/{PROJECT_ID}/locations/{REGION}/models/{MODEL_ID}"
model = aiplatform.Model(model_resource_name)

batch_job = model.batch_predict(
    job_display_name='boston-housing-test-batch',
    gcs_source=INPUT_URI,
    gcs_destination_prefix=OUTPUT_PREFIX,
    machine_type="n1-standard-4"
)

batch_job.wait()
print("Batch prediction completed. Check output in:", OUTPUT_PREFIX)
