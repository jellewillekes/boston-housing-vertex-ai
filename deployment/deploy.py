import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
from models.algorithms.ada_boost_q import AdaBoostQ

# Configs
PROJECT_ID = "affor-models"
DATASET_X = "affor-data.vertex_baseline.baseline_x"
DATASET_Y = "affor-data.vertex_baseline.baseline_y"
BUCKET_NAME = "test-baseline"
MODEL_PATH = "models/trained/model/adaboostq_trained.json"


def load_data(date='2012-01-01'):
    """Load data from BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)

    query_x = f"SELECT * FROM `{DATASET_X}` WHERE DATE(date) >= '{date}'"
    query_y = f"SELECT * FROM `{DATASET_Y}` WHERE DATE(date) >= '{date}'"

    X = client.query(query_x).to_dataframe()
    y = client.query(query_y).to_dataframe()

    X['date'] = pd.to_datetime(X['date'])
    y['date'] = pd.to_datetime(y['date'])

    X = X.set_index(['date', 'sec_id'])
    y = y.set_index(['date', 'sec_id'])

    # Convert label column to CategoricalDtype
    if 'label' in y.columns:
        categories = [-1, 0, 1]
        y['label'] = pd.Categorical(y['label'], categories=categories, ordered=True)
        y = y['label']

    return X, y


def train_model(date):
    """Train the AdaBoostQ model and save it to GCS."""
    print(f"Training model for data after date: {date}")

    # Load data
    X, y = load_data(date)
    print("Data loaded from BigQuery.")

    # Train model
    model = AdaBoostQ(layers=30)
    model.fit(X, y)
    print("Model training complete.")

    # Save model to GCS
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH)

    with blob.open("w") as f:
        model.save(f)

    print(f"Model saved to GCS at: gs://{BUCKET_NAME}/{MODEL_PATH}")


if __name__ == "__main__":
    train_model("2012-01-01")
