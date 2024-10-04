from keras._tf_keras.keras.applications import VGG16
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Flatten, Dropout
from keras._tf_keras.keras.optimizers import Adam

# Load the VGG16 model pre-trained on ImageNet, excluding the top layers
def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers of the pre-trained model to avoid training them
    for layer in base_model.layers:
        layer.trainable = False

    # Build the full model by adding custom layers on top of the pre-trained base model
    model = Sequential([
        base_model,
        Flatten(),  # Flatten the output of the VGG16 model
        Dense(128, activation='relu'),
        Dropout(0.5),  # Prevent overfitting
        Dense(1, activation='sigmoid')  # Binary classification layer
    ])

    # Compile the model with Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model