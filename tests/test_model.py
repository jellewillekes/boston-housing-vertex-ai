import pytest
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from training.model import feed_forward_net


@pytest.fixture
def boston_data():
    # Load Boston Housing dataset from TensorFlow
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()
    # Combine train and test datasets for more control
    X = tf.concat([X_train, X_test], axis=0).numpy()
    y = tf.concat([y_train, y_test], axis=0).numpy()
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_model_training(boston_data):
    X_train, X_test, y_train, y_test = boston_data

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the model
    model = feed_forward_net(input_shape=(X_train.shape[1],))
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Evaluate the model
    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    # Assert that MAE is within a reasonable range
    assert mae < 20, f"Model MAE too high: {mae}"
