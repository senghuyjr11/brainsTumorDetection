from data_loader import DataLoader
from cnn_model import create_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import numpy as np
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

# Create the CNN model
model = create_model()

# Callbacks for early stopping and learning rate reduction
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
]

# Train the CNN model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    class_weight=class_weights,
    callbacks=callbacks
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Save the model
model.save('brain_tumor_cnn_model.keras')

# Generate predictions and evaluate
y_pred = (model.predict(test_generator, steps=len(test_generator)) > 0.5).astype(int)
y_true = test_generator.classes

# Display confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['No Tumor', 'Tumor']))
