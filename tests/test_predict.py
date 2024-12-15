import unittest
import numpy as np
import tensorflow as tf
import os
import json
from fastapi.testclient import TestClient
from serving.predict import (
    validate_and_transform_payload,
    process_jsonl_input,
    predict_instances,
    app,
)


class TestPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the actual Keras model from the local_model_dir."""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(root_dir, "local_model_dir", "model.keras")

        try:
            cls.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the model from {model_path}: {e}")

        cls.client = TestClient(app)

    def setUp(self):
        """Set up test data with correct shape."""
        self.valid_data = {"instances": [[1.0] * 13, [2.0] * 13]}  # Two valid instances
        self.invalid_data_no_instances = {"invalid_key": [[1.0] * 13]}
        self.invalid_data_shape = {"instances": [[1.0] * 5]}  # Invalid shape
        self.valid_jsonl_data = (
            '{"instances": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]}\n'
            '{"instances": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]}'
        )

    # Unit tests for utility functions
    def test_validate_and_transform_payload_success(self):
        """Test successful payload validation and transformation."""
        instances = validate_and_transform_payload(self.valid_data)
        self.assertIsInstance(instances, np.ndarray)
        self.assertEqual(instances.shape, (2, 13))

    def test_validate_and_transform_payload_failure_no_instances(self):
        """Test failure when 'instances' key is missing."""
        with self.assertRaises(ValueError):
            validate_and_transform_payload(self.invalid_data_no_instances)

    def test_validate_and_transform_payload_failure_wrong_shape(self):
        """Test failure when instances have the wrong shape."""
        with self.assertRaises(ValueError):
            validate_and_transform_payload(self.invalid_data_shape)

    def test_process_jsonl_input_success(self):
        """Test successful JSONL input processing."""
        instances = process_jsonl_input(self.valid_jsonl_data)  # JSONL string
        self.assertEqual(len(instances), 2)
        self.assertEqual(len(instances[0]), 13)

    def test_predict_instances_success(self):
        """Test successful predictions using the actual model."""
        instances = np.array([[1.0] * 13, [2.0] * 13])
        predictions = predict_instances(instances, self.model)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)

    def test_predict_instances_model_not_loaded(self):
        """Test prediction failure due to missing model."""
        instances = np.array([[1.0] * 13])
        with self.assertRaises(ValueError):
            predict_instances(instances, None)

    # API endpoint tests
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})  # Updated to match FastAPI response

    def test_predict_endpoint_json(self):
        """Test the predict endpoint with standard JSON payload."""
        response = self.client.post(
            "/predict",
            json=self.valid_data,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("predictions", response_data)
        self.assertEqual(len(response_data["predictions"]), len(self.valid_data["instances"]))

    def test_predict_endpoint_jsonl(self):
        """Test the predict endpoint with JSONL payload."""
        response = self.client.post(
            "/predict",
            data=self.valid_jsonl_data,
            headers={"Content-Type": "application/jsonlines"},
        )
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("predictions", response_data)
        self.assertEqual(len(response_data["predictions"]), 2)

    def test_predict_endpoint_invalid_json(self):
        """Test the predict endpoint with invalid JSON payload."""
        response = self.client.post(
            "/predict",
            json=self.invalid_data_no_instances,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        response_data = response.json()
        self.assertIn("detail", response_data)

    def test_predict_endpoint_invalid_jsonl(self):
        """Test the predict endpoint with invalid JSONL payload."""
        invalid_jsonl = """
        {"wrong_key": [1.0, 2.0, 3.0]}
        {"instances": [1.0, 2.0]}
        """
        response = self.client.post(
            "/predict",
            data=invalid_jsonl,
            headers={"Content-Type": "application/jsonlines"},
        )
        self.assertEqual(response.status_code, 400)
        response_data = response.json()
        self.assertIn("detail", response_data)

    def test_predict_with_jsonl_file(self):
        """Test predictions using the content of prediction_input.jsonl file."""
        # Step 1: Load the JSONL file content
        file_path = os.path.join(os.path.dirname(__file__), "../prediction_input.jsonl")
        with open(file_path, "r") as file:
            jsonl_content = file.read()

        # Step 2: Send the content as a POST request to the /predict endpoint
        response = self.client.post(
            "/predict",
            data=jsonl_content,
            headers={"Content-Type": "application/jsonlines"}
        )

        # Step 3: Assertions
        self.assertEqual(response.status_code, 200)  # Ensure response is OK
        response_data = response.json()
        self.assertIn("predictions", response_data)
        self.assertEqual(len(response_data["predictions"]), 4)
        print("Predictions:", response_data["predictions"])


if __name__ == "__main__":
    unittest.main()
