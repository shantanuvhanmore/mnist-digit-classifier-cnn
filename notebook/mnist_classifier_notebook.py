#!/usr/bin/env python
# coding: utf-8

# # MNIST Digit Classification with CNN
# 
# This notebook demonstrates how to build and train a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) from the MNIST dataset using TensorFlow/Keras.
# 
# - **Dataset:** MNIST (28x28 grayscale images)
# - **Tools:** Python, TensorFlow/Keras, Matplotlib
# - **Goal:** Achieve high accuracy in digit recognition
# 

# ## 1. Import Required Libraries and Load Dataset
# 

# In[1]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Training data shape:', x_train.shape)
print('Test data shape:', x_test.shape)


# ## 2. Preprocess Data
# 

# In[2]:


# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN (add channel dimension)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print('Reshaped training data:', x_train.shape)


# ## 3. Define CNN Model Architecture
# 

# In[4]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes for digits
])

model.summary()


# ## 4. Compile and Train Model
# 

# In[5]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)


# ## 5. Evaluate Model Performance
# 

# In[6]:


test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Accuracy:', test_acc)


# ## 6. Visualize Training Accuracy
# 

# In[7]:


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.savefig('../outputs/accuracy_plot.png')
plt.show()


# ## 7. Make Predictions and Display Sample Results
# 

# In[8]:


import numpy as np

predictions = model.predict(x_test)

def show_sample(i):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}")
    plt.axis('off')
    plt.savefig(f'../outputs/sample_prediction_{i}.png')
    plt.show()

show_sample(10)  # Try different indices

