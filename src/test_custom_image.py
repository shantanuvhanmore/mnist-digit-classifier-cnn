import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

# Usage: python src/test_custom_image.py path_to_image.png
if len(sys.argv) < 2:
    print("Usage: python src/test_custom_image.py path_to_image.png")
    sys.exit(1)

image_path = sys.argv[1]

# Load and preprocess the image
img = Image.open(image_path).convert('L').resize((28, 28))
img_array = np.array(img) / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

# Load trained model
model = tf.keras.models.load_model('outputs/mnist_cnn_model.keras')
pred = model.predict(img_array)

plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {np.argmax(pred)}")
plt.axis('off')
plt.show()
