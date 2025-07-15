from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np 

# Define the input layer
input_layer = Input(shape=(20,))

# Add hidden layers with batch normalization
hidden_layer1 = Dense(64, activation='relu')(input_layer)
batch_norm1 = BatchNormalization()(hidden_layer1)
hidden_layer2 = Dense(64, activation='relu')(batch_norm1)
batch_norm2 = BatchNormalization()(hidden_layer2)

# Define the output layer
output_layer = Dense(1, activation='sigmoid')(batch_norm2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Example data (in practice, use real dataset) 

X_train = np.random.rand(1000, 20) 
y_train = np.random.randint(2, size=(1000, 1)) 
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
# Example test data
X_test = np.random.rand(200, 20) 
y_test = np.random.randint(2, size=(200, 1)) 

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')