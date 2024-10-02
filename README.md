# Brain Tumor Detection using CNN and Hybrid Classifiers

This project uses a pre-trained Convolutional Neural Network (CNN) for feature extraction from MRI images, followed by multiple machine learning classifiers to detect brain tumors. The CNN extracts features from the MRI images, and classifiers like SVM, RandomForest, XGBoost, KNN, and Logistic Regression are used to classify the images as "Tumor" or "No Tumor."

## Features
- **CNN for Feature Extraction**: Pre-trained CNN model (`brain_tumor_cnn_model.keras`) extracts features from MRI images.
- **Hybrid Classifiers**: Multiple classifiers (SVM, XGBoost, RandomForest, KNN, and Logistic Regression) trained on extracted features.
- **Image Prediction**: Ability to predict on new MRI images or batch of images in a directory.

## Project Structure
├── dataset/                 # Train, val, test datasets
├── models/                  # Saved models
├── cnn_model.py             # CNN model definition
├── data_loader.py           # Data loading utilities
├── main.py                  # CNN model training
├── hybrid_classifier.py     # ML classifiers on CNN features
├── predict.py               # Prediction script
└── requirements.txt         # Dependencies

## Requirements
- Python 3.x
- Required libraries are listed in `requirements.txt`.
- numpy
- opencv-python
- scikit-learn
- tensorflow
- keras
- xgboost
- lightgbm

### Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### Installation
1. Clone the repository:
- git clone https://github.com/your-username/brain-tumor-detection.git
- cd brain-tumor-detection
- pip install -r requirements.txt
- pip install numpy opencv-python scikit-learn tensorflow keras xgboost lightgbm

### Train CNN Model
- python main.py # The model will be saved as brain_tumor_cnn_model.keras

### Hybrid Classifier
- python hybrid_classifier.py # We use GridSearchCV to optimize models like XGBoost and Random Forest

Evaluation Results
Classifiers are evaluated using accuracy, confusion matrix, and classification report. Latest results:
SVM Test Accuracy: 62.75%
XGBoost Test Accuracy: 66.67%
Random Forest Test Accuracy: 64.71%
KNN Test Accuracy: 74.51%
Logistic Regression Test Accuracy: 62.75%

### Prediction
Predict on new MRI images using:
- python predict.py --image_path dataset/test/no/5_no.jpg

### This project is licensed under the MIT License.
https://senghuyjr11.surge.sh/