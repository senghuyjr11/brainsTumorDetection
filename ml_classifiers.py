from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from keras.models import load_model, Model
import numpy as np
from data_loader import DataLoader

# Load the pre-trained CNN model
model = load_model('brain_tumor_cnn_model.keras')

def extract_features_from_cnn(model, generator):
    """
    Extract features from the CNN model using the Flatten or GlobalAveragePooling2D layer.
    """
    features = []
    labels = []

    for batch, label_batch in generator:
        batch_features = model.predict(batch)
        features.append(batch_features)
        labels.append(label_batch)

        # Stop when all data is processed
        if len(labels) >= len(generator):
            break

    # Stack features and labels into numpy arrays
    features = np.vstack(features)
    labels = np.hstack(labels)

    return features, labels

def train_classifiers(train_features, train_labels, test_features, test_labels):
    """
    Train multiple classifiers on the extracted features.
    """

    # Hyperparameter tuning for SVM using GridSearchCV
    print("Running hyperparameter tuning for SVM...")
    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(train_features, train_labels)

    print("Best parameters for SVM:", grid.best_params_)
    print("Best accuracy for SVM on training data:", grid.best_score_)

    # Train and test SVM with the best parameters from grid search
    svm_best = grid.best_estimator_
    svm_pred = svm_best.predict(test_features)
    print('SVM Accuracy on test set:', accuracy_score(test_labels, svm_pred))

    # Train and test Random Forest
    rf = RandomForestClassifier()
    rf.fit(train_features, train_labels)
    rf_pred = rf.predict(test_features)
    print('Random Forest Accuracy:', accuracy_score(test_labels, rf_pred))

    # Train and test KNN
    knn = KNeighborsClassifier()
    knn.fit(train_features, train_labels)
    knn_pred = knn.predict(test_features)
    print('KNN Accuracy:', accuracy_score(test_labels, knn_pred))

    # Train and test Logistic Regression
    lr = LogisticRegression()
    lr.fit(train_features, train_labels)
    lr_pred = lr.predict(test_features)
    print('Logistic Regression Accuracy:', accuracy_score(test_labels, lr_pred))

if __name__ == "__main__":
    # Step 1: Define dataset directories
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    test_dir = 'dataset/test'

    # Step 2: Create an instance of DataLoader and get the data generators
    data_loader = DataLoader(train_dir, val_dir, test_dir)
    train_generator, val_generator, test_generator = data_loader.get_data_generators()

    # Step 3: Extract features from the CNN
    print("Extracting features from CNN model for training set...")
    train_features, train_labels = extract_features_from_cnn(model, train_generator)

    print("Extracting features from CNN model for test set...")
    test_features, test_labels = extract_features_from_cnn(model, test_generator)

    # Step 4: Train classifiers and print their accuracies
    print("Training classifiers on extracted features...")
    train_classifiers(train_features, train_labels, test_features, test_labels)
