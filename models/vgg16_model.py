from keras._tf_keras.keras.applications import VGG16
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import RMSprop
from keras._tf_keras.keras.regularizers import l2


# Load the VGG16 model pre-trained on ImageNet, excluding the top layers
def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the initial layers of the pre-trained model to avoid training them
    # Allow the last few layers to be trainable for fine-tuning
    for layer in base_model.layers[:-8]:  # Unfreeze more layers for fine-tuning
        layer.trainable = True

    # Build the full model by adding custom layers on top of the pre-trained base model
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # Prevent overfitting
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # Prevent overfitting
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification layer

    model = Model(inputs, outputs)

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