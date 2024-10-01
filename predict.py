import numpy as np
import cv2
from keras._tf_keras.keras.models import load_model

# Load the saved model
model = load_model('brain_tumor_cnn_model.keras')

# Preprocess a single image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read the image using OpenCV
    img = cv2.resize(img, (224, 224))  # Resize to match the model input size
    img = img / 255.0  # Normalize pixel values to 0-1
    img = np.expand_dims(img, axis=0)  # Add batch dimension (for model input)
    return img

# Function to predict on a new image
def predict_on_new_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)

    # Output the prediction result
    if prediction > 0.5:
        print("Prediction: Tumor Detected")
    else:
        print("Prediction: No Tumor")

# Example usage: Predicting on a new MRI image
if __name__ == "__main__":
    # Path to your new MRI image
    new_image_path = 'dataset/train/no/7 no.jpg'

    # Run the prediction
    predict_on_new_image(new_image_path)
