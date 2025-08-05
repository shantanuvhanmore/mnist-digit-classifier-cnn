cd c:\codes\mnist-digit-classifier-cnn
.venv\Scripts\activatecd c:\codes\mnist-digit-classifier-cnn
.venv\Scripts\activatecd c:\codes\mnist-digit-classifier-cnn
.venv\Scripts\activatecd c:\codes\mnist-digit-classifier-cnn
.venv\Scripts\activate# MNIST Digit Classifier using CNN 🧠

This project uses a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.

## Features
- 98%+ accuracy on test set
- Simple CNN using Keras
- Visual output and evaluation

## Dataset
- MNIST: 60,000 training and 10,000 test images of digits (0–9)

## How to Run
1. Clone repo
2. Install requirements
3. Run `notebook/mnist_classifier_notebook.ipynb`

## How to Run (Windows CMD)

1. Open the main project folder in Command Prompt (CMD).
2. Create a virtual environment:
   ```cmd
   python -m venv .venv
   ```
3. Activate the virtual environment:
   ```cmd
   .venv\Scripts\activate
   ```
4. Install requirements:
   ```cmd
   pip install -r requirements.txt
   ```
5. Train and save the model:
   ```cmd
   python src/train.py
   ```
6. Test with your own image (see instructions below):
   ```cmd
   python src/test_custom_image.py data/nine2.png
   ```

## Test with Your Own Image

1. Prepare your image:
   - Make sure the image is square (e.g., 28x28 pixels).
   - The digit should be between 0–9.
   - The background should be black.
   - You can use Paint or any image editor for this.
   - **Place your image in the `data` folder.**
   - If you have difficulty creating an image, you can use one of the sample images already provided in the `data` folder.

2. Run the test script with your image:
   ```
   python src/test_custom_image.py data/your_image.png
   ```
   - Example:
     ```
     python src/test_custom_image.py data/nine2.png
     ```

## Sample Output
![Accuracy Plot](outputs/accuracy_plot.png)

![sample prediction 0](outputs\sample_prediction_0.png)

## Model Save Name
- After running `src/train.py`, the trained model will be saved as `mnist_cnn_model.keras` in the `outputs` folder.

## Author
Your Name
