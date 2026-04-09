import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sample dataset
X = np.random.rand(100, 10)
y = np.random.randint(2, size=(100, 1))

# Build CNN Model
model = keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(X, y, epochs=5)

print("Model Training Completed")
