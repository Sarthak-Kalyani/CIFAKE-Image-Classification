# CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images

An AI-based image classification system that distinguishes between REAL
and AI-generated FAKE images using a Convolutional Neural Network (CNN)
and provides visual explanations using Grad-CAM.

## Features

- REAL / FAKE image classification
- CNN-based image classification
- Grad-CAM explainability
- Flask web interface
- Image preprocessing using OpenCV
- Model evaluation
- Confusion matrix
- Training and validation graphs

## Dataset

The project uses the CIFAKE dataset.

Processed dataset:
- 100,000 images
- 50,000 REAL
- 50,000 FAKE
- 80,000 training images
- 20,000 testing images

The dataset itself is not included in this repository.

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 95.93% |
| Precision | 94.70% |
| Recall | 97.29% |
| F1-Score | 95.98% |

## Architecture

Input: 32 × 32 × 3

Conv2D → MaxPooling → Dropout  
→ Conv2D → MaxPooling → Dropout  
→ Flatten → Dense → Softmax

## Explainability

Grad-CAM is used to generate a heatmap showing regions
associated with the CNN's prediction.

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Flask
- SQLite
- HTML/CSS

## Running the Project

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

```
## Live Demo

https://cifake-image-classification.onrender.com
