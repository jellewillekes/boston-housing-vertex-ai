from kfp import compiler
from google.cloud import aiplatform
from pipelines.boston_pipeline import boston_pipeline

# Set your environment variables
PROJECT_ID = "affor-models"
REGION = "europe-west1"
BUCKET = "gs://boston-example"

# Define paths and names
PIPELINE_ROOT = f"{BUCKET}/pipeline_root"
PIPELINE_JSON = "boston_pipeline.json"
PIPELINE_DISPLAY_NAME = "boston-housing-pipeline"


def compile_pipeline():
    """
  Compiles the pipeline into a JSON file.
  """
    print("Compiling the pipeline...")
    compiler.Compiler().compile(
        pipeline_func=boston_pipeline,
        package_path=PIPELINE_JSON,
    )
    print(f"Pipeline compiled and saved to: {PIPELINE_JSON}")


def submit_pipeline():
    """
  Submits the compiled pipeline to Vertex AI.
  """
    print("Submitting the pipeline job to Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=REGION)

    pipeline_job = aiplatform.PipelineJob(
        display_name=PIPELINE_DISPLAY_NAME,
        template_path=PIPELINE_JSON,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project_id": PROJECT_ID,
            "region": REGION,
        },
    )
    pipeline_job.run(sync=True)
    print("Pipeline submitted successfully.")


if __name__ == "__main__":
    compile_pipeline()
    submit_pipeline()
