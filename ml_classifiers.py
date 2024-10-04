import numpy as np
from keras._tf_keras.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils.config import CNN_MODEL
from utils.data_loader import DataLoader

# Load the pre-trained CNN model
model = load_model('models/'+CNN_MODEL)

def extract_features_from_cnn(model, generator):
    features, labels = [], []

    # Calculate the number of batches to process
    steps = len(generator)

    for i in range(steps):
        batch, label_batch = next(generator)
        batch_features = model.predict(batch)
        features.append(batch_features)
        labels.append(label_batch)

    return np.vstack(features), np.hstack(labels)


def train_classifiers(train_features, train_labels, test_features, test_labels):
    # Hyperparameter tuning for SVM
    param_grid_svm = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    grid_svm = GridSearchCV(SVC(), param_grid_svm, refit=True, verbose=0)
    grid_svm.fit(train_features, train_labels)

    print(f"Best SVM Params: {grid_svm.best_params_}, Best Training Accuracy: {grid_svm.best_score_:.4f}")
    svm_best = grid_svm.best_estimator_

    # Train Logistic Regression and Random Forest separately
    lr = LogisticRegression()
    lr.fit(train_features, train_labels)

    rf = RandomForestClassifier()
    rf.fit(train_features, train_labels)

    # Ensemble model: SVM, Logistic Regression, and Random Forest
    ensemble_model = VotingClassifier(estimators=[('svm', svm_best), ('lr', lr), ('rf', rf)], voting='hard')
    ensemble_model.fit(train_features, train_labels)
    print(f"Ensemble Accuracy: {accuracy_score(test_labels, ensemble_model.predict(test_features)):.4f}")

    # Hyperparameter tuning for Random Forest
    param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
    grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, refit=True, verbose=0)
    grid_rf.fit(train_features, train_labels)
    print(f"Best Random Forest Params: {grid_rf.best_params_}, Best Training Accuracy: {grid_rf.best_score_:.4f}")

    # Train and evaluate KNN
    knn = KNeighborsClassifier()
    knn.fit(train_features, train_labels)
    print(f'KNN Accuracy: {accuracy_score(test_labels, knn.predict(test_features)):.4f}')

    # Evaluate Logistic Regression (already trained for VotingClassifier)
    print(f'Logistic Regression Accuracy: {accuracy_score(test_labels, lr.predict(test_features)):.4f}')

if __name__ == "__main__":
    # Dataset directories
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    test_dir = 'dataset/test'

    # Load the dataset
    data_loader = DataLoader(train_dir, val_dir, test_dir)
    train_generator, val_generator, test_generator = data_loader.get_data_generators()

    # Extract features from CNN
    print("Extracting CNN features for training set...")
    train_features, train_labels = extract_features_from_cnn(model, train_generator)

    print("Extracting CNN features for test set...")
    test_features, test_labels = extract_features_from_cnn(model, test_generator)

    # Train and evaluate classifiers
    print("Training classifiers...")
    train_classifiers(train_features, train_labels, test_features, test_labels)
