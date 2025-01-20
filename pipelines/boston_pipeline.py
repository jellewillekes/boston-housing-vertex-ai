from kfp import dsl
from kfp.dsl import ContainerSpec

from kfp.dsl import (Dataset, Output, Input, Model)


# 1) Load Data
@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "numpy==1.26.4",
        "tensorflow==2.17.0"
    ]
)
def load_data(output_data: Output[Dataset]):
    """
    Loads data (in real scenario, from GCS/BigQuery),
    saves as .npz for the training container.
    """
    import numpy as np
    import tensorflow as tf

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

    # Save to a local .npz file
    npz_path = f"{output_data.path}.npz"  # .path is a directory, so we add .npz
    np.savez(npz_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # This is important so the next step can see the actual file
    output_data.metadata["npz_file"] = npz_path


# 2) Train Model (container step)
@dsl.container_component
def train_model(
    data: Input[Dataset],
    output_model: Output[Model],
    project_id: str,
    model_dir: str = "artifacts"
):
    return ContainerSpec(
        image=f"europe-west1-docker.pkg.dev/{project_id}/boston-example/boston-training-image:latest",
        command=["python", "train.py"],
        args=[
            "--model_dir", model_dir,
            "--data_path", data.metadata["npz_file"],
        ],
    )


# 3) Deploy Model
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform==1.71.1"]
)
def deploy_model(
    project_id: str,
    region: str,
    model_dir: str,
    endpoint_display_name: str = "boston-housing",
    deployed_model_display_name: str = "boston-housing",
):
    """
    Uploads the model to Vertex AI and deploys it to an endpoint.
    The container to serve predictions is in your 'boston-serving-image'.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    # 3a) Upload the model (assuming model artifacts in model_dir)
    serving_image_uri = f"{region}-docker.pkg.dev/{project_id}/boston-example/boston-serving-image:latest"
    model = aiplatform.Model.upload(
        display_name="boston-housing",
        artifact_uri=model_dir,
        serving_container_image_uri=serving_image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )
    print(f"Uploaded model: {model.resource_name}")

    # 3b) Deploy to Endpoint
    endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_display_name}")
    if endpoints:
        endpoint = endpoints[0]
        print(f"Using existing endpoint: {endpoint.display_name}")
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
        print(f"Created endpoint: {endpoint_display_name}")

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        machine_type="n1-standard-4",
        traffic_split={"0": 100},
    )
    print(f"Model deployed to endpoint: {endpoint.name}")


# 4) Define Pipeline
@dsl.pipeline(name="boston-housing-pipeline")
def boston_pipeline(project_id: str = "affor-models", region: str = "europe-west1"):
    # Step A: load data
    data_task = load_data()

    # Step B: train model (container step)
    train_task = train_model(
        data=data_task.outputs["output_data"],
        project_id=project_id,
        model_dir="gs://boston-example/artifacts",
    )

    # Step C: deploy model
    deploy_task = deploy_model(
        project_id=project_id,
        region=region,
        model_dir="gs://boston-example/artifacts",
    )
    deploy_task.after(train_task)
