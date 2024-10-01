from data_loader import DataLoader
from cnn_model import create_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import numpy as np  # Import numpy for array handling
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau

train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# Step 1: Create an instance of DataLoader to load the dataset
data_loader = DataLoader(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32)

# Get the train, validation, and test data generators
train_generator, val_generator, test_generator = data_loader.get_data_generators()

# Step 2: Create the CNN model
model = create_model()

# Step 3: Address class imbalance using class weights
# Compute class weights based on training data
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_generator.classes
)

# Convert class weights array to dictionary
class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}  # Create a dictionary for class weights

# Step 4: Add EarlyStopping and ReduceLROnPlateau callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Step 5: Train the model with class weights and callbacks
print("Starting model training...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,  # Can change the number of epochs
    validation_data=val_generator,
    validation_steps=len(val_generator),
    class_weight=class_weights,  # Apply class weights as a dictionary
    callbacks=[early_stopping, reduce_lr]  # Apply callbacks
)

# Step 6: Evaluate the model on the test set
print("Evaluating model on the test set...")
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Optional: Save the trained model for future use
model.save('brain_tumor_cnn_model.keras')
print("Model saved as 'brain_tumor_cnn_model.keras'")

# Generate predictions for the test data
y_pred = model.predict(test_generator, steps=len(test_generator))
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary output

# Get true labels from the test generator
y_true = test_generator.classes

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Classification Report (includes precision, recall, F1-score)
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['No Tumor', 'Tumor']))
