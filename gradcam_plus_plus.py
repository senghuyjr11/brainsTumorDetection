import cv2
import numpy as np
import tensorflow as tf
from keras import Model
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
from models.resnet50_model import create_resnet_model

# Create the model
model_resnet = create_resnet_model(input_shape=(150, 150, 3))

def process_image(img_path, target_size=(150, 150)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def compute_iou(binary_heatmap, ground_truth):
    intersection = np.sum(np.logical_and(binary_heatmap, ground_truth))
    union = np.sum(np.logical_or(binary_heatmap, ground_truth))
    iou = intersection / union if union != 0 else 0
    return iou

def compute_dice(binary_heatmap, ground_truth):
    intersection = np.sum(np.logical_and(binary_heatmap, ground_truth))
    dice_score = (2 * intersection) / (np.sum(binary_heatmap) + np.sum(ground_truth))
    return dice_score

# Example evaluation on a test set of images
def evaluate_gradcam_on_test_set(test_images_and_masks, threshold=0.5):
    iou_scores = []
    dice_scores = []

    for img_path, ground_truth_path in test_images_and_masks:
        # Load and process the image
        img_array_resnet = process_image(img_path)

        # Load the ground-truth mask
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

        # Check if the ground-truth mask was loaded correctly
        if ground_truth is None:
            print(f"Error: Ground truth mask not found or cannot be loaded: {ground_truth_path}")
            continue

        # Resize the ground-truth mask to the same size as the input image
        ground_truth = cv2.resize(ground_truth, (150, 150))
        ground_truth = np.where(ground_truth > 128, 1, 0)  # Convert to binary mask

        # Generate Grad-CAM++ heatmap
        heatmap_resnet = get_gradcam_plus_plus(model_resnet, img_array_resnet, 'conv5_block3_out')

        # Reduce multi-channel heatmap to a single channel by taking the mean or max
        heatmap_resnet = np.mean(heatmap_resnet, axis=-1)  # Alternatively, use np.max(heatmap_resnet, axis=-1)

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
        plt.title("Grad-CAM++ (ResNet50)")
        plt.imshow(superimposed_img_resnet)
        plt.show()

    # Calculate average IoU and Dice scores
    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)

    print(f"Average IoU: {avg_iou}")
    print(f"Average Dice Score: {avg_dice}")
    return avg_iou, avg_dice


def get_gradcam_plus_plus(model, img_array, last_conv_layer_name):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    # Compute the gradients of the loss with respect to the output feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Compute positive gradients, ReLU-like behavior
    grads = tf.where(grads > 0, grads, 0.0)

    # Compute squared gradients
    grads_squared = grads ** 2

    # Compute third order gradients
    grads_cubed = grads_squared * grads

    # Sum over the gradients of each feature map along batch, height, and width
    sum_grads = tf.reduce_sum(grads, axis=(0, 1, 2))

    # Get the number of channels
    channels = conv_outputs.shape[-1]

    # Ensure sum_grads is correctly shaped
    sum_grads = tf.reshape(sum_grads, [1, 1, channels])

    # Tile sum_grads to match the spatial dimensions of grads_squared
    sum_grads = tf.tile(sum_grads, [conv_outputs.shape[1], conv_outputs.shape[2], 1])  # Tile across height and width

    # Compute alpha values (importance of each feature map)
    alpha_num = grads_squared
    alpha_denom = 2 * grads_squared + sum_grads
    alpha = alpha_num / (alpha_denom + tf.keras.backend.epsilon())

    # Compute the importance weights for each feature map
    weights = tf.reduce_sum(alpha * tf.maximum(grads, 0), axis=(0, 1))

    # Compute the weighted combination of the feature maps
    weighted_conv_outputs = tf.reduce_sum(weights * conv_outputs, axis=-1)

    # Normalize the heatmap
    if np.max(weighted_conv_outputs) != 0:
        heatmap = weighted_conv_outputs / np.max(weighted_conv_outputs)
    else:
        heatmap = weighted_conv_outputs  # Fallback in case of zero max value

    return heatmap.numpy()


def superimpose_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Ensure the heatmap is 2D (single-channel)
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, axis=-1)

    # Normalize the heatmap between 0 and 255
    heatmap = np.uint8(255 * heatmap)

    # Ensure the heatmap is single-channel before applying the colormap
    if len(heatmap.shape) == 3 and heatmap.shape[-1] != 1:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    # Apply the JET color map to the heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = np.clip(heatmap * alpha + img, 0, 255).astype(np.uint8)

    return superimposed_img


# Example usage of Grad-CAM++
if __name__ == '__main__':

    # Example test set (replace with actual image and mask paths)
    test_images_and_masks = [
        ('dataset/test/yes/Y11.jpg', 'dataset/mask/mask_Y11.jpg')
    ]
    # Run evaluation on the test set
    evaluate_gradcam_on_test_set(test_images_and_masks)

