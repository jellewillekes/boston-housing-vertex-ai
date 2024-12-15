# scripts/batch_predict.py
from google.cloud import aiplatform
from load_config import PROJECT_ID, REGION, BUCKET

aiplatform.init(project=PROJECT_ID, location=REGION)

# After uploading the model, copy its resource name or MODEL_ID.
# For full automation, you'd store the MODEL_ID after upload.
# Let's say we provide MODEL_ID as an environment variable or you can hard-code here.
# Alternatively, if you want it all automated, you can retrieve the model by display name.
# Here, assume you know the MODEL_ID from the upload_model.py output.

MODEL_ID = "2666082600891711488"  # Replace after upload step
INPUT_URI = f"{BUCKET}input/prediction_input.jsonl"  # Just adjust path if needed
OUTPUT_PREFIX = f"{BUCKET}output/"

model_resource_name = f"projects/{PROJECT_ID}/locations/{REGION}/models/{MODEL_ID}"
model = aiplatform.Model(model_resource_name)

batch_job = model.batch_predict(
    gcs_source=INPUT_URI,
    gcs_destination_prefix=OUTPUT_PREFIX,
    machine_type="n1-standard-4"
)

batch_job.wait()
print("Batch prediction completed. Check output in:", OUTPUT_PREFIX)
