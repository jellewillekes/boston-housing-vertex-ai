import requests

# Prediction endpoint
predict_url = "http://127.0.0.1:8080/predict"

try:
    # Read the JSONL file and send a POST request to the /predict endpoint
    with open("prediction_input.jsonl", "r") as file:
        response = requests.post(
            predict_url,
            data=file,
            headers={"Content-Type": "application/jsonlines"}
        )
        # Print the response
        if response.status_code == 200:
            print("Prediction successful!")
            print(response.json())
        else:
            print(f"Prediction failed: {response.status_code}, {response.json()}")
except FileNotFoundError:
    print("Error: The file 'prediction_input.jsonl' was not found.")
except Exception as e:
    print(f"Error during prediction request: {e}")
