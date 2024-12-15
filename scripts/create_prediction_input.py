import tensorflow as tf
import json
import os


# Load Boston Housing dataset
def load_boston_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()
    return X_test


# Save observations to JSONL
def save_to_jsonl(data, output_file):
    with open(output_file, 'w') as f:
        for row in data:
            json.dump({"input": row.tolist()}, f)  # Convert array to list for JSON serialization
            f.write('\n')


def main():
    data = load_boston_data()

    # Select 4 observations for prediction
    selected_data = data[:4]

    # Save to JSONL in the root folder
    script_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.abspath(os.path.join(script_folder, ".."))
    output_file = os.path.join(root_folder, "prediction_input.jsonl")
    save_to_jsonl(selected_data, output_file)

    print(f"Saved 4 observations for prediction to {output_file}")


if __name__ == "__main__":
    main()
