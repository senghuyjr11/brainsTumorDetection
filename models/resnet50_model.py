from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras._tf_keras.keras.models import Model

def create_resnet_model(input_shape):
    # Load the pre-trained ResNet50 model
    base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers initially for transfer learning
    for layer in base_model_resnet.layers:
        layer.trainable = False

    # Add custom classification layers
    x_resnet = base_model_resnet.output
    x_resnet = GlobalAveragePooling2D()(x_resnet)  # Use GAP instead of Flatten
    x_resnet = Dense(512, activation='relu')(x_resnet)
    x_resnet = Dropout(0.5)(x_resnet)
    output_resnet = Dense(1, activation='sigmoid')(x_resnet)

    # Create the model
    model_resnet = Model(inputs=base_model_resnet.input, outputs=output_resnet)

    # Compile the model
    model_resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model_resnet