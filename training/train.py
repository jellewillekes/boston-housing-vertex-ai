# training/train.py

import argparse
import os
import numpy as np
import tensorflow as tf
from model import feed_forward_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Local or GCS path to save the model")
    parser.add_argument("--data_path", type=str,
                        help="Local or GCS path to .npz file with X_train, y_train, X_test, y_test")
    args = parser.parse_args()

    # Load data
    if args.data_path:
        print(f"Loading data from: {args.data_path}")
        # If data_path is on GCS, the container should have gcsfuse or you need to download it
        # For local example, just assume it's a local .npz
        data = np.load(args.data_path)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
    else:
        print("No data_path provided. Using built-in Boston Housing dataset.")
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

    input_shape = (X_train.shape[1],)

    # Create and train model
    model = feed_forward_net(input_shape)
    model.fit(X_train, y_train, epochs=5, validation_split=0.1)

    # Evaluate
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {mae}")

    # Ensure the output directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(args.model_dir, "model.keras")
    model.save(model_path)
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    main()
