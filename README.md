# 👁️ Gender Eye Classification CNN

<p align="center">
  <!-- <img src="docs/images/cover.png" alt="Gender Eye Classification Banner" width="100%"> -->
  <img width="2752" height="1536" alt="cover" src="https://github.com/user-attachments/assets/6373789f-47b0-40f9-a585-53fb5a8736db" />

</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Jupyter%20Notebook-orange?style=flat-square&logo=jupyter" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-FF6F00?style=flat-square&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Language-Python-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Institution-RAIN%20RDA-purple?style=flat-square" />
</p>

## The Challenge

If you look at a full human face, determining gender is usually instant. But what happens if you strip away the jawline, the hair, the nose, and the facial structure? 

This project was built to answer a specific computer vision challenge given during my deep learning examination at Robotics and Artificial Intelligence Nigeria (RAIN). The goal was to build a Convolutional Neural Network from scratch that can look at a tightly cropped, 60x60 pixel image of a human eye and accurately predict the gender. 

This is a raw feature extraction problem. The network has to learn to identify subtle geometric and textural differences in eyelash density, tear ducts, and skin folds that the human brain normally glosses over.

## 📊 The Dataset & Preprocessing

The model was trained on the `eye_gender_data` dataset. Before feeding anything into the neural network, the raw data pipeline handles several crucial formatting steps.

<p align="center">
  <!-- <img src="docs/images/dataset_sample.png" alt="Sample of the dataset showing labeled eyes" width="80%"> -->
  <img width="1189" height="224" alt="dataset_sample" src="https://github.com/user-attachments/assets/f1af51c5-dd9e-4621-b338-5f1682b81f2a" />

</p>
<p align="center"><i>Sample output directly from the training notebook</i></p>

*   **Total Images:** 9,220 images
*   **Data Split:** 6,454 images for training, and 2,766 held back for strict testing. Validation during training was set to a 20 percent split of the training data.
*   **Resizing:** Every image is uniformly compressed to 60x60 pixels using OpenCV.
*   **Normalization:** Pixel arrays are divided by 255.0 to scale them between 0 and 1, ensuring stable and fast gradient descent.
*   **Encoding:** Labels are mapped to binary integers where **Male = 1** and **Female = 0**.

## 🧠 Model Architecture

Instead of relying on heavy pre-trained models, I engineered a custom Sequential CNN using Keras. The architecture is designed to progressively extract higher-level features without overfitting.

| Layer Type | Details | Purpose |
| :--- | :--- | :--- |
| **Input** | `(60, 60, 3)` | Accepts the resized RGB image arrays. |
| **Conv2D + MaxPool** | 32 filters, 3x3, ReLU | Extracts low-level features like edges and basic shapes. |
| **Conv2D + MaxPool** | 64 filters, 3x3, ReLU | Extracts mid-level features like iris curvature. |
| **Conv2D + MaxPool** | 128 filters, 3x3, ReLU | Extracts high-level, complex geometric patterns. |
| **Flatten** | 3200 units | Converts the 2D feature maps into a 1D vector. |
| **Dense** | 128 units, ReLU | Fully connected layer for pattern classification. |
| **Dropout** | Rate = 0.5 | Randomly drops 50% of neurons to prevent dataset memorization. |
| **Output (Dense)** | 1 unit, Sigmoid | Outputs a probability score between 0 and 1. |

*Total Trainable Parameters: 503,105*

## 📈 Training & Performance

The model was compiled using the `Adam` optimizer and `binary_crossentropy` loss. It was trained over 20 epochs.

<p align="center">
<img width="1189" height="390" alt="training_graphs" src="https://github.com/user-attachments/assets/506a5863-569d-468f-a70e-8c31074e19bd" /></p>

The network learned the underlying patterns highly effectively. When evaluated against the 2,766 unseen test images, the final metrics were:

*   **Test Accuracy:** 90.67%
*   **Test Loss:** 0.3917

## 💻 How to Run This Project

You do not need to retrain the model from scratch to test it. I have exported the final trained weights.

### Prerequisites
Make sure you have the following installed in your Python environment:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
```

### Running the Code
1. Clone this repository:

```bash
git clone https://github.com/Salimahh/human-eye-classification-model.git
cd human-eye-classification-model
```

2. To view the full training pipeline, open the Jupyter Notebook:

```bash
jupyter notebook "gender eye classification model.ipynb"
```

3. To use the pre-trained model for your own predictions, load the included Keras file:

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model("gender_eye_classification_model.keras")

# Preprocess your image
img = cv2.imread("your_image.jpg")
img = cv2.resize(img, (60, 60))
img = img / 255.0
img = np.expand_dims(img, axis=0) # Add batch dimension

# Predict
prediction = model.predict(img)
if prediction >= 0.5:
    print("Predicted: Male")
else:
    print("Predicted: Female")
```
