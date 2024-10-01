import numpy as np
import cv2
from keras._tf_keras.keras.models import load_model
import os

# Load the saved model
model = load_model('brain_tumor_cnn_model.keras')

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

# Function to predict on a new image
def predict_on_new_image(image_path, threshold=0.5):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        prediction = model.predict(preprocessed_image)[0][0]  # Get the first (and only) prediction value

        # Print predicted probability
        print(f"Predicted probability: {prediction:.4f}")

        # Output the classification result
        if prediction > threshold:
            print("Prediction: Tumor Detected")
        else:
            print("Prediction: No Tumor")
    else:
        print(f"Prediction could not be made for {image_path}")

# Function to predict on a batch of images in a directory
def predict_on_directory(directory_path, threshold=0.5):
    for filename in os.listdir(directory_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory_path, filename)
            print(f"\nProcessing: {image_path}")
            predict_on_new_image(image_path, threshold)

# Example usage: Predicting on a new MRI image
if __name__ == "__main__":
    # Single image path for prediction
    new_image_path = 'dataset/train/no/5 no.jpg'

    # Run the prediction on a single image
    predict_on_new_image(new_image_path)

    # Optionally, you can predict on all images in a directory
    # directory_path = 'path_to_your_images'
    # predict_on_directory(directory_path)
