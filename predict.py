import os
import cv2
import joblib
import numpy as np
from keras._tf_keras.keras.models import load_model

from utils.config import VGG16_MODEL

# Load the pre-trained CNN/VGG16 model as a feature extractor
model_path = os.path.join('models',VGG16_MODEL)
if os.path.exists(model_path):
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    print("Model loaded successfully.")
else:
    raise ValueError(f"File not found: {model_path}. Please ensure the file exists and is accessible.")

# Load the trained PCA transformer
pkl_dir = 'saved_classifiers'
pca_path = os.path.join(pkl_dir, 'best_pca.pkl')
if os.path.exists(pca_path):
    pca = joblib.load(pca_path)
    print("PCA transformer loaded successfully.")
else:
    raise ValueError(f"File not found: {pca_path}. Please ensure the file exists and is accessible.")

# Load the trained ensemble model
ensemble_model_path = os.path.join(pkl_dir, 'ensemble_classifier.pkl')
if os.path.exists(ensemble_model_path):
    ensemble_classifier = joblib.load(ensemble_model_path)
    print("Ensemble classifier loaded successfully.")
else:
    raise ValueError(f"File not found: {ensemble_model_path}. Please ensure the file exists and is accessible.")

# Preprocess a single image
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to load image from {image_path}")
        img = cv2.resize(img, (224, 224))  # Resize to model input size
        img = img / 255.0  # Normalize pixel values to 0-1
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


"""
# Function to predict on a new image using the cnn model
def predict_on_new_image_cnn(image_path, threshold=0.5):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        # Get the first (and only) prediction value
        prediction = model.predict(preprocessed_image)[0][0]

        # Print predicted probability
        print(f"Predicted probability: {prediction:.4f}")

        # Output the classification result
        if prediction > threshold:
            print("Prediction: Tumor Detected")
        else:
            print("Prediction: No Tumor")
    else:
        print(f"Prediction could not be made for {image_path}")
"""

# Function to predict on a new image using the hybrid classifier
def predict_on_new_image_classifier(image_path):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        try:
            # Extract features using the VGG16 model
            features = model.predict(preprocessed_image, verbose=0)
            print(f"Features shape: {features.shape}")

            # Apply PCA transformation to the extracted features
            features_pca = pca.transform(features.reshape(1, -1))
            print(f"Features shape after PCA: {features_pca.shape}")

            # Predict using the loaded ensemble classifier
            probabilities = ensemble_classifier.predict_proba(features_pca)
            tumor_probability = probabilities[0][1]  # Probability of the 'Tumor' class

            # Set a threshold to determine the label
            if tumor_probability > 0.5:
                predicted_label = 'Tumor'
            else:
                predicted_label = 'No Tumor'

            print(f"Prediction: {predicted_label} (Probability: {tumor_probability:.4f})")
        except Exception as e:
            print(f"Error during prediction: {e}")
    else:
        print(f"Prediction could not be made for {image_path}")

# Function to predict on a batch of images in a directory
def predict_on_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory_path, filename)
            print(f"\nProcessing: {image_path}")
            predict_on_new_image_classifier(image_path)

# Example usage: Predicting on a new MRI image
if __name__ == "__main__":
    # Single image path for prediction
    new_image_path = 'dataset/val/no/18 no.jpg'

    # Run the prediction on a single image
    predict_on_new_image_classifier(new_image_path)

    # Optionally, you can predict on all images in a directory
    # directory_path = 'path_to_your_images'
    # predict_on_directory(directory_path)