import tensorflow as tf
from tensorflow.keras.datasets import mnist
from model import build_model
import matplotlib.pyplot as plt
import numpy as np
import os

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

output_dir = os.path.join(os.path.dirname(__file__), '../outputs')
model.save(os.path.join(output_dir, 'mnist_cnn_model.keras'))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
