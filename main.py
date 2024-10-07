import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight

from models.resnet50_model import create_resnet_model
from models.vgg16_model import get_callbacks
from utils.config import RESNET50_MODEL
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
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.labels), y=train_generator.labels)
class_weights = dict(enumerate(class_weights))

# Load the VGG16 model pre-trained on ImageNet, excluding the top layers
model = create_resnet_model(input_shape=(224, 224, 3))
callbacks = get_callbacks()

# Fit the model with updated steps and callbacks
model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Save the trained model
model.save('models/'+RESNET50_MODEL)

# Generate predictions and evaluate
y_pred = (model.predict(test_generator, steps=len(test_generator)) > 0.5).astype(int)
y_true = test_generator.classes

# Display confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['No Tumor', 'Tumor']))
