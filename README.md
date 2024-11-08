# Kidney Disease Classification Using Deep Learning

This project focuses on classifying kidney disease using medical images. It uses deep learning models, specifically **VGG16** and a custom **Convolutional Neural Network (CNN)**, to predict kidney disease. The dataset was sourced from Kaggle and consists of labeled kidney images.

## Project Overview

- **Dataset**: [Kidney Disease Dataset on Kaggle](https://www.kaggle.com/)
  - Labeled images: `Healthy` and `Diseased`
  - Image format: `.jpg` or `.png`
- **Models**:
  - **VGG16**: Fine-tuned pre-trained model with ImageNet weights.
  - **Custom CNN**: A tailored Convolutional Neural Network designed for this task.
- **Tools**: TensorFlow, Keras, NumPy, Pandas, Matplotlib.

## Project Files

- `data/`: Contains the kidney image dataset (train, validation, test splits).
- `models/`: Contains the saved models (`vgg16_model.h5` and `custom_cnn_model.h5`).
- `notebooks/`: Jupyter notebooks for data preprocessing and model training (`data_preprocessing.ipynb`, `model_training.ipynb`).
- `src/`: Python scripts for model definition (`model.py`) and helper functions (`utils.py`).
- `predict.py`: Script for making predictions on new images.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.

## Installation

To set up the project, clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd kidney-disease-classification
pip install -r requirements.txt
