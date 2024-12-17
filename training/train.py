import argparse
import os
import tensorflow as tf
from model import feed_forward_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Local or cloud path to save the model")
    args = parser.parse_args()

    # Load dataset
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
