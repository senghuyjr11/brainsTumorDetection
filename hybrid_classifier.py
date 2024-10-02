import numpy as np
from keras._tf_keras.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data_loader import DataLoader  # Assuming you have a custom DataLoader

# Load the pre-trained CNN model
cnn_model = load_model('brain_tumor_cnn_model.keras')

# Extract features from the CNN model
def extract_features_from_cnn(model, generator):
    features, labels = [], []
    steps = len(generator)

    for i in range(steps):
        batch, label_batch = next(generator)
        batch_features = model.predict(batch)
        features.append(batch_features)
        labels.append(label_batch)

    return np.vstack(features), np.hstack(labels)

# Train and evaluate multiple classifiers
def train_hybrid_classifiers(train_features, train_labels, test_features, test_labels):
    # SVM with GridSearchCV
    param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid_svm = GridSearchCV(SVC(), param_grid_svm, refit=True, verbose=0)
    grid_svm.fit(train_features, train_labels)
    print(f"Best SVM Params: {grid_svm.best_params_}")
    svm_pred = grid_svm.best_estimator_.predict(test_features)
    print(f"SVM Test Accuracy: {accuracy_score(test_labels, svm_pred):.4f}")

    # XGBoost
    xgb = XGBClassifier()
    xgb.fit(train_features, train_labels)
    xgb_pred = xgb.predict(test_features)
    print(f"XGBoost Test Accuracy: {accuracy_score(test_labels, xgb_pred):.4f}")

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(train_features, train_labels)
    rf_pred = rf.predict(test_features)
    print(f"Random Forest Test Accuracy: {accuracy_score(test_labels, rf_pred):.4f}")

    # KNN
    knn = KNeighborsClassifier()
    knn.fit(train_features, train_labels)
    knn_pred = knn.predict(test_features)
    print(f"KNN Test Accuracy: {accuracy_score(test_labels, knn_pred):.4f}")

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(train_features, train_labels)
    lr_pred = lr.predict(test_features)
    print(f"Logistic Regression Test Accuracy: {accuracy_score(test_labels, lr_pred):.4f}")

if __name__ == "__main__":
    # Dataset directories
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    test_dir = 'dataset/test'

    # Load the dataset using the custom DataLoader
    data_loader = DataLoader(train_dir, val_dir, test_dir)
    train_generator, val_generator, test_generator = data_loader.get_data_generators()

    # Extract CNN features
    train_features, train_labels = extract_features_from_cnn(cnn_model, train_generator)
    test_features, test_labels = extract_features_from_cnn(cnn_model, test_generator)

    # Train and evaluate classifiers
    train_hybrid_classifiers(train_features, train_labels, test_features, test_labels)
