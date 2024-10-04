import os

import joblib
import numpy as np
from keras._tf_keras.keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from utils.config import VGG16_MODEL
from utils.data_loader import DataLoader  # Assuming you have a custom DataLoader

# Load the pre-trained VGG16 model
model = load_model(VGG16_MODEL)

# Create a directory to store the .pkl files if it doesn't exist
pkl_dir = 'saved_classifiers'
if not os.path.exists(pkl_dir):
    os.makedirs(pkl_dir)

# Extract features from the VGG16 model
def extract_features_from_vgg16(model, generator):
    features, labels = [], []
    steps = len(generator)

    for i in range(steps):
        batch, label_batch = next(generator)
        batch_features = model.predict(batch)
        features.append(batch_features)
        labels.append(label_batch)

    return np.vstack(features), np.hstack(labels)

# Train and evaluate multiple classifiers using an ensemble approach
def train_hybrid_classifiers_with_ensemble(train_features, train_labels, test_features, test_labels):
    # Apply PCA to reduce dimensionality
    param_grid_pca = {'n_components': [10, 20, 30, 50, 100]}  # Grid for tuning n_components
    best_pca = None
    best_accuracy = 0
    best_train_features_pca = None
    best_test_features_pca = None

    # Loop through different n_components to find the best value
    for n in param_grid_pca['n_components']:
        pca = PCA(n_components=min(n, train_features.shape[1]))
        train_features_pca = pca.fit_transform(train_features)
        test_features_pca = pca.transform(test_features)

        # Using SVM to test the performance for each value of n_components
        svm = SVC(C=1, kernel='linear')
        svm.fit(train_features_pca, train_labels)
        svm_pred = svm.predict(test_features_pca)
        accuracy = accuracy_score(test_labels, svm_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_pca = pca
            best_train_features_pca = train_features_pca
            best_test_features_pca = test_features_pca

    print(f"Best PCA n_components: {best_pca.n_components_}")

    # Use the best PCA for training all classifiers
    train_features_pca = best_train_features_pca
    test_features_pca = best_test_features_pca

    # Define individual classifiers
    svm = SVC(C=10, kernel='rbf', probability=True)
    xgb = XGBClassifier()
    rf = RandomForestClassifier()
    knn = KNeighborsClassifier()
    lr = LogisticRegression()

    # Train the classifiers and print their results
    svm.fit(train_features_pca, train_labels)
    svm_pred = svm.predict(test_features_pca)
    print(f"SVM Test Accuracy: {accuracy_score(test_labels, svm_pred):.4f}")
    joblib.dump(svm, os.path.join(pkl_dir, 'svm_classifier.pkl'))  # Save the trained SVM model

    xgb.fit(train_features_pca, train_labels)
    xgb_pred = xgb.predict(test_features_pca)
    print(f"XGBoost Test Accuracy: {accuracy_score(test_labels, xgb_pred):.4f}")
    joblib.dump(xgb, os.path.join(pkl_dir, 'xgb_classifier.pkl'))  # Save the trained XGBoost model

    rf.fit(train_features_pca, train_labels)
    rf_pred = rf.predict(test_features_pca)
    print(f"Random Forest Test Accuracy: {accuracy_score(test_labels, rf_pred):.4f}")
    joblib.dump(rf, os.path.join(pkl_dir, 'rf_classifier.pkl'))  # Save the trained Random Forest model

    knn.fit(train_features_pca, train_labels)
    knn_pred = knn.predict(test_features_pca)
    print(f"KNN Test Accuracy: {accuracy_score(test_labels, knn_pred):.4f}")
    joblib.dump(knn, os.path.join(pkl_dir, 'knn_classifier.pkl'))  # Save the trained KNN model

    lr.fit(train_features_pca, train_labels)
    lr_pred = lr.predict(test_features_pca)
    print(f"Logistic Regression Test Accuracy: {accuracy_score(test_labels, lr_pred):.4f}")
    joblib.dump(lr, os.path.join(pkl_dir, 'lr_classifier.pkl'))  # Save the trained Logistic Regression model

    # Voting Classifier (Ensemble)
    ensemble = VotingClassifier(estimators=[
        ('svm', svm),
        ('xgb', xgb),
        ('rf', rf),
        ('knn', knn),
        ('lr', lr)
    ], voting='soft')

    # Train the ensemble model
    ensemble.fit(train_features_pca, train_labels)
    ensemble_pred = ensemble.predict(test_features_pca)
    print(f"Ensemble Test Accuracy: {accuracy_score(test_labels, ensemble_pred):.4f}")

    # Save the ensemble model
    joblib.dump(ensemble, os.path.join(pkl_dir, 'ensemble_classifier.pkl'))
    # Save the PCA transformer
    joblib.dump(best_pca, os.path.join(pkl_dir, 'best_pca.pkl'))

if __name__ == "__main__":
    # Dataset directories
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    test_dir = 'dataset/test'

    # Load the dataset using the custom DataLoader
    data_loader = DataLoader(train_dir, val_dir, test_dir)
    train_generator, val_generator, test_generator = data_loader.get_data_generators()

    # Extract VGG16 features
    train_features, train_labels = extract_features_from_vgg16(model, train_generator)
    test_features, test_labels = extract_features_from_vgg16(model, test_generator)

    # Train and evaluate classifiers using an ensemble approach
    train_hybrid_classifiers_with_ensemble(train_features, train_labels, test_features, test_labels)