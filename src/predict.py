import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.models.load_model('outputs/mnist_cnn_model.h5')
predictions = model.predict(x_test)

def show_sample(i):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}")
    plt.axis('off')
    plt.savefig(f'outputs/sample_prediction_{i}.png')
    plt.show()

show_sample(10)
