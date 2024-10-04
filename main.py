import numpy as np
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

from models.cnn_model import create_cnn_model
from models.vgg16_model import create_vgg16_model
from utils.config import VGG16_MODEL, CNN_MODEL
from utils.data_loader import DataLoader

# Define dataset directories
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# Load the dataset
data_loader = DataLoader(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32)
train_generator, val_generator, test_generator = data_loader.get_data_generators()

# Compute class weights for handling class imbalance
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_generator.labels
)
class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}

# Load the VGG16 model pre-trained on ImageNet, excluding the top layers
model = create_vgg16_model()

# Calculating steps_per_epoch and validation_steps
steps_per_epoch = len(train_generator) if hasattr(train_generator, '__len__') else train_generator.samples // train_generator.batch_size
validation_steps = len(val_generator) if hasattr(val_generator, '__len__') else val_generator.samples // val_generator.batch_size

# Updated callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),  # Update to monitor available metric
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, min_lr=1e-5)  # Update to monitor available metric
]

# Fit the model with updated steps and callbacks
model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=val_generator,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=callbacks
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Save the trained model
model.save('models/'+VGG16_MODEL)

# Generate predictions and evaluate
y_pred = (model.predict(test_generator, steps=len(test_generator)) > 0.5).astype(int)
y_true = test_generator.classes

# Display confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['No Tumor', 'Tumor']))
