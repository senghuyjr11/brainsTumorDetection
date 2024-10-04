# Brain Tumor Detection using CNN and Hybrid Classifiers

This project uses a pre-trained Convolutional Neural Network (CNN) for feature extraction from MRI images, followed by multiple machine learning classifiers to detect brain tumors. The CNN extracts features from the MRI images, and classifiers like SVM, RandomForest, XGBoost, KNN, and Logistic Regression are used to classify the images as "Tumor" or "No Tumor."

## Features
- **CNN for Feature Extraction**: Pre-trained CNN model (`brain_tumor_vgg16_model.keras` or `brain_tumor_cnn_model.keras`) extracts features from MRI images.
- **Hybrid Classifiers**: Multiple classifiers (SVM, XGBoost, RandomForest, KNN, and Logistic Regression) trained on extracted features.
- **Ensemble Classifier**: Combines the predictions of individual classifiers to improve accuracy.
- **Image Prediction**: Ability to predict on new MRI images or a batch of images in a directory.

## Project Structure
```
├── dataset/ # Train, val, test datasets
│   ├── test/
│   ├── train/
│   └── val/
├── models/ # Saved models
│   ├── brain_tumor_cnn_model.keras
│   ├── brain_tumor_vgg16_model.keras
│   ├── cnn_model.py
│   └── vgg16_model.py
├── saved_classifiers/ # Saved machine learning classifiers
│   ├── best_pca.pkl
│   ├── ensemble_classifier.pkl
│   ├── knn_classifier.pkl
│   ├── lr_classifier.pkl
│   ├── rf_classifier.pkl
│   ├── svm_classifier.pkl
│   └── xgb_classifier.pkl
├── utils/ # Utility scripts
│   ├── config.py
│   ├── data_loader.py
│   └── split_dataset.py
├── classifiers/ # Classifier training scripts
│   ├── hybrid_classifier.py
│   └── ml_classifiers.py
├── predict.py # Prediction script using hybrid classifiers
├── main.py # CNN model training script
└── requirements.txt # Dependencies
```

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
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/senghuyjr11/brainsTumorDetection.git
   cd brainsTumorDetection
   pip install -r requirements.txt
   pip install numpy opencv-python scikit-learn tensorflow keras xgboost lightgbm
   ```

### Train CNN Model
- Train the CNN model using:
  ```sh
  python main.py
  ```
  The model will be saved as `brain_tumor_cnn_model.keras` or `brain_tumor_vgg16_model.keras`.

### Hybrid Classifier
- Train hybrid classifiers using extracted CNN features:
  ```sh
  python classifiers/hybrid_classifier.py
  ```
  We use `GridSearchCV` to optimize models like XGBoost and Random Forest.

### Evaluation Results
Classifiers are evaluated using accuracy, confusion matrix, and classification report. Latest results:
- **SVM Test Accuracy**: 76.47%
- **XGBoost Test Accuracy**: 70.59%
- **Random Forest Test Accuracy**: 68.63%
- **KNN Test Accuracy**: 76.47%
- **Logistic Regression Test Accuracy**: 60.78%
- **Ensemble Test Accuracy**: 76.47%

### Prediction
Predict on new MRI images using the hybrid classifier:
- Predict a single image:
  ```sh
  python predict.py --image_path dataset/test/no/5_no.jpg
  ```
- Predict a batch of images in a directory:
  ```sh
  python predict.py --directory_path path_to_your_images
  ```

### License
This project is licensed under the MIT License.

https://senghuyjr11.surge.sh/