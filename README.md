````
# 🧠 MNIST Digit Classifier using CNN

This project uses a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) from the MNIST dataset. It achieves over **98% accuracy** and is a great starting point to understand image classification using deep learning.

---

## 📌 Features

- ✅ Over 98% accuracy on the MNIST test set  
- 🧱 Built using **TensorFlow** and **Keras**  
- 📊 Visual training output and predictions  
- 💾 Saves trained model to `outputs/` directory  
- 🧪 Test your own digit images

---

## 📦 Dataset

> **MNIST** – 70,000 grayscale images of handwritten digits  
- 60,000 training images  
- 10,000 test images  
- Image size: 28x28 pixels

---
## 🖥️ How to Run

### 🔧 Setup (Windows CMD)

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/mnist-digit-classifier-cnn.git
   cd mnist-digit-classifier-cnn

2. **Create a virtual environment:**

   ```cmd
   python -m venv .venv
   ```

3. **Activate the virtual environment:**

   ```cmd
   .venv\Scripts\activate
   ```

4. **Install dependencies:**

   ```cmd
   pip install -r requirements.txt
   ```

5. **Train the model:**

   ```cmd
   python src/train.py
   ```

6. **Test with a custom image:**

   ```cmd
   python src/test_custom_image.py data/nine2.png
   ```

---

## 🧪 Test with Your Own Image

1. Prepare your image:

   * Size: **28x28 pixels**
   * Format: `.png` recommended
   * Black background, white digit (0–9)
   * Place it inside the `data/` folder

2. Run the test:

   ```bash
   python src/test_custom_image.py data/your_image.png
   ```

   Example:

   ```bash
   python src/test_custom_image.py data/nine2.png
   ```

---

## 📈 Sample Output

<p float="left">
  <img src="outputs/accuracy_plot.png" width="300" />
  <img src="outputs/sample_prediction_0.png" width="300" />
</p>

---

## 💾 Model Output

* Trained model saved as:

  ```
  outputs/mnist_cnn_model.keras
  ```

---

## 📁 Project Structure

```
mnist-digit-classifier-cnn/
│
├── src/
│   ├── train.py                # Training script
│   └── test_custom_image.py   # Test custom digit images
│
├── data/                      # Sample or custom input images
├── output/                    # Saved models, plots, predictions
├── notebook/                  # Jupyter notebook version
│   └── mnist_classifier_notebook.ipynb
│
├── requirements.txt           # Required Python packages
└── README.md                  # Project documentation
```

---

## 👨‍💻 Author

**Shantanu Vhanmore**
Feel free to fork, contribute, or reach out with questions!

---

## ⭐️ Star This Repo

If you found this useful, consider giving it a ⭐️ on GitHub — it helps others find it!

```