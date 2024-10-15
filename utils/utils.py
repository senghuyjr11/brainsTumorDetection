import cv2
import numpy as np
from keras._tf_keras.keras.applications.resnet50 import preprocess_input  # Import the ResNet50 preprocessing function

def preprocess_image(image_path, target_size=(150, 150)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to load image from {image_path}")

        img = cv2.resize(img, target_size)  # Resize to target size
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Apply the ResNet50 preprocessing (scales pixel values to the range expected by the model)
        img = preprocess_input(img)

        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def compute_iou(binary_heatmap, ground_truth):
    intersection = np.sum(np.logical_and(binary_heatmap, ground_truth))
    union = np.sum(np.logical_or(binary_heatmap, ground_truth))
    iou = intersection / union if union != 0 else 0
    return iou

def compute_dice(binary_heatmap, ground_truth):
    intersection = np.sum(np.logical_and(binary_heatmap, ground_truth))
    dice_score = (2 * intersection) / (np.sum(binary_heatmap) + np.sum(ground_truth))
    return dice_score