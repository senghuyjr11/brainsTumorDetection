import cv2
import numpy as np
import tensorflow as tf
from keras import Model
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
from models.resnet50_model import create_resnet_model
from utils.utils import preprocess_image, compute_iou, compute_dice

# Create the model
model_resnet = create_resnet_model(input_shape=(150, 150, 3))


# Example evaluation on a test set of images
def evaluate_gradcam_on_test_set(test_images_and_masks, threshold=0.5):
    iou_scores = []
    dice_scores = []

    for img_path, ground_truth_path in test_images_and_masks:
        # Load and process the image
        img_array_resnet = preprocess_image(img_path)

        # Load the ground-truth mask
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

        # Check if the ground-truth mask was loaded correctly
        if ground_truth is None:
            print(f"Error: Ground truth mask not found or cannot be loaded: {ground_truth_path}")
            continue

        # Resize the ground-truth mask to the same size as the input image
        ground_truth = cv2.resize(ground_truth, (150, 150))
        ground_truth = np.where(ground_truth > 128, 1, 0)  # Convert to binary mask

        # Generate Grad-CAM heatmap
        heatmap_resnet = get_gradcam_resnet(model_resnet, img_array_resnet, 'conv5_block3_out')

        # Resize the heatmap to match the input image size
        heatmap_resnet = cv2.resize(heatmap_resnet, (150, 150))

        # Threshold the heatmap to binary mask
        binary_heatmap = np.where(heatmap_resnet >= threshold, 1, 0)

        # Compute IoU and Dice for the image
        iou = compute_iou(binary_heatmap, ground_truth)
        dice = compute_dice(binary_heatmap, ground_truth)

        iou_scores.append(iou)
        dice_scores.append(dice)

        # Display the result (Optional)
        superimposed_img_resnet = superimpose_gradcam(img_path, heatmap_resnet)
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(ground_truth, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Grad-CAM (ResNet50)")
        plt.imshow(superimposed_img_resnet)
        plt.show()

    # Calculate average IoU and Dice scores
    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)

    print(f"Average IoU: {avg_iou}")
    print(f"Average Dice Score: {avg_dice}")
    return avg_iou, avg_dice


def get_gradcam_resnet(model, img_array, last_conv_layer_name):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    # Compute the gradients of the loss with respect to the last convolutional layer
    grads = tape.gradient(loss, conv_outputs)

    # Compute the mean of the gradients for each feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each feature map by the importance weights
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def superimpose_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert the heatmap to an RGB image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = np.clip(heatmap * alpha + img, 0, 255).astype(np.uint8)

    return superimposed_img

if __name__ == '__main__':
    # Select an image for visualization
    img_path_resnet = '/home/senghuyjr11/Projects/brainsTumorDetection/dataset/new/Y3.jpg'
    img_resnet = load_img(img_path_resnet, target_size=(150, 150))
    img_array_resnet = img_to_array(img_resnet)
    img_array_resnet = np.expand_dims(img_array_resnet, axis=0)

    # Generate Grad-CAM heatmap using the final convolutional layer in ResNet50
    heatmap_resnet = get_gradcam_resnet(model_resnet, img_array_resnet, 'conv5_block3_out')

    # Superimpose the heatmap on the original image
    superimposed_img_resnet = superimpose_gradcam(img_path_resnet, heatmap_resnet)

    # Example test set (replace with actual image and mask paths)
    test_images_and_masks = [
        ('dataset/test/yes/Y11.jpg', 'dataset/mask/mask_Y11.jpg'),
        ('dataset/test/yes/Y24.jpg', 'dataset/mask/mask_Y24.jpg'),
        # Add more test images and corresponding masks here
    ]

    # Run evaluation
    evaluate_gradcam_on_test_set(test_images_and_masks)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_resnet)

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM (ResNet50)")
    plt.imshow(superimposed_img_resnet)
    plt.show()
