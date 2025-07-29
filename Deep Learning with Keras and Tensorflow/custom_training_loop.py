import os
import warnings
import tensorflow as tf 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.callbacks import Callback
import numpy as np

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Set TensorFlow log level to suppress warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Step 1: Set Up the Environment
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
x_train, x_test = x_train / 255.0, x_test / 255.0 
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)



model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10)
])

# Step 3: Define Loss Function and Optimizer

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
optimizer = tf.keras.optimizers.Adam()

# Step 4: Implement the Custom Training Loop

epochs = 2
# train_dataset = train_dataset.repeat(epochs)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
for epoch in range(epochs):
    print(f'Start of epoch {epoch + 1}')

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)  # Forward pass
            loss_value = loss_fn(y_batch_train, logits)  # Compute loss

        # Compute gradients and update weights
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Logging the loss every 200 steps
        if step % 200 == 0:
            print(f'Epoch {epoch + 1} Step {step}: Loss = {loss_value.numpy()}')

import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 

# Step 1: Set Up the Environment
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0 

# Create a batched dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)



model = Sequential([ 
    Flatten(input_shape=(28, 28)),  # Flatten the input to a 1D vector
    Dense(128, activation='relu'),  # First hidden layer with 128 neurons and ReLU activation
    Dense(10)  # Output layer with 10 neurons for the 10 classes (digits 0-9)
])

