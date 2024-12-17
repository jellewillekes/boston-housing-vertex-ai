import tensorflow as tf


def feed_forward_net(input_shape):
    # Simple feed forward neural net with regression output
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),  # ReLU for non-linearity
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Regression output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
