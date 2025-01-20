# run_boston_pipeline.py
import os
import sys
from kfp import compiler
from google.cloud import aiplatform

# Import your pipeline function
from pipelines.boston_pipeline import boston_pipeline


def compile_and_submit_pipeline(
    project_id: str,
    region: str,
    bucket_name: str,
    pipeline_name: str = "boston-housing-pipeline",
):
    """Compiles your KFP pipeline to JSON, then submits it to Vertex AI Pipelines."""

    # 1) Compile Pipeline to JSON
    pipeline_json = f"{pipeline_name}.json"
    print("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=boston_pipeline,
        package_path=pipeline_json,
    )
    print(f"Compiled pipeline: {pipeline_json}")

    # 2) Submit Pipeline to Vertex AI
    #    Use aiplatform.PipelineJob to submit the compiled pipeline JSON
    pipeline_root = f"gs://{bucket_name}/pipeline_root"
    aiplatform.init(project=project_id, location=region)

    print("Submitting pipeline to Vertex AI Pipelines...")
    job = aiplatform.PipelineJob(
        display_name=pipeline_name,
        template_path=pipeline_json,
        pipeline_root=pipeline_root,
        parameter_values={
            "project_id": project_id,
            "region": region,
            # Add more parameters if your pipeline_func has more
        },
    )
    job.run(sync=True)  # or sync=False to return immediately
    print("Pipeline submitted. Check Vertex AI Pipelines UI for status.")


if __name__ == "__main__":
    # Simple CLI interface
    project_id = os.getenv("PROJECT_ID", "affor-models")
    region = os.getenv("REGION", "europe-west1")
    bucket_name = os.getenv("BUCKET_NAME", "boston-example")

    compile_and_submit_pipeline(project_id, region, bucket_name)
