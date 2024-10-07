import cv2
import numpy as np
import tensorflow as tf
from keras import Model
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt

from models.resnet50_model import create_resnet_model

#Create the model
model_resnet = create_resnet_model(input_shape=(299, 299, 3))
# model_resnet = create_vgg16_model(input_shape=(150, 150, 3))

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
    img_path_resnet = 'dataset/train/yes/Y13.jpg'
    img_resnet = load_img(img_path_resnet, target_size=(299, 299))
    img_array_resnet = img_to_array(img_resnet)
    img_array_resnet = np.expand_dims(img_array_resnet, axis=0)

    # Generate Grad-CAM heatmap using the final convolutional layer in ResNet50
    # VGG16 model (block5_conv3), resnet model (conv5_block3_out)
    heatmap_resnet = get_gradcam_resnet(model_resnet, img_array_resnet, 'conv5_block3_out')  # ResNet50's last conv block

    # Superimpose the heatmap on the original image
    superimposed_img_resnet = superimpose_gradcam(img_path_resnet, heatmap_resnet)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_resnet)

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM (ResNet50)")
    plt.imshow(superimposed_img_resnet)
    plt.show()