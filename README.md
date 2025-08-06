````markdown
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
   git clone https://github.com/shantanuvhanmore/mnist-digit-classifier-cnn.git
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

* Accuracy Plot:
<img width="640" height="480" alt="accuracy_plot" src="https://github.com/user-attachments/assets/a48f643f-7b27-4e8f-ae80-0f3c0a9272d5" />


* Sample Prediction:
<img width="640" height="480" alt="sample_prediction" src="https://github.com/user-attachments/assets/acad91dd-19a4-4e61-827d-6709745b04d9" />


* Test with Custom Images:

    #1. with ACCURATE PREDICTION
  
   <img width="1366" height="768" alt="Screenshot 2025-08-05 114826" src="https://github.com/user-attachments/assets/2600bea8-1925-4fd3-b99c-008033dafcec" />
   
   <img width="1366" height="768" alt="Screenshot 2025-08-05 114725" src="https://github.com/user-attachments/assets/adcc1b6c-1083-470c-a4dd-9cae4742f560" />

    *2. with INACCURATE PREDICTION
       
   <img width="1366" height="768" alt="Screenshot 2025-08-05 114803" src="https://github.com/user-attachments/assets/ace3295c-5a0b-4aae-8f08-aa0f861e3fbd" />
   
   <img width="1366" height="768" alt="Screenshot 2025-08-05 115157" src="https://github.com/user-attachments/assets/8e8988dd-56d8-4dc3-aa40-8a626a9077f6" />


---

## 💾 Model Output

* After running the src/train.py Trained model saved as:

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
