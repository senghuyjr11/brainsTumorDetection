from keras._tf_keras.keras.applications import VGG16
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.layers import GlobalAveragePooling2D
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import RMSprop


# Load the VGG16 model pre-trained on ImageNet, excluding the top layers
def create_vgg16_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the initial layers of the pre-trained model to avoid training them
    # Allow the last few layers to be trainable for fine-tuning
    for layer in base_model.layers:
        layer.trainable = False

    x_resnet = base_model.output
    x_resnet = GlobalAveragePooling2D()(x_resnet)  # Use GAP instead of Flatten
    x_resnet = Dense(512, activation='relu')(x_resnet)
    x_resnet = Dropout(0.5)(x_resnet)
    output_resnet = Dense(1, activation='sigmoid')(x_resnet)
    model = Model(inputs=base_model.input, outputs=output_resnet)

    # Compile the model with Adam optimizer
    model.compile(optimizer=RMSprop(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Define callbacks for early stopping and learning rate reduction
def get_callbacks():
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    return [early_stopping, reduce_lr]