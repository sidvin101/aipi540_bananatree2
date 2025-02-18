import os
import numpy as np
import joblib
import cv2
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# data
dataset_path = "banana_tree"

def extract_features(image_path):
    """extract classical ML features"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG Features
    hog_features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)

    # LBP Features
    lbp = local_binary_pattern(gray, P=8, R=2) 
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 256), density=True) 

    # Color Histogram Features
    color_hist = []
    for i in range(3): 
        hist = cv2.calcHist([img], [i], None, [16], [0, 256]) 
        color_hist.extend(hist.flatten())

    features = np.hstack([hog_features, lbp_hist, color_hist])
    return features

X, y, image_paths = [], [], []

class_labels = {label: idx for idx, label in enumerate(os.listdir(dataset_path)) if os.path.isdir(os.path.join(dataset_path, label))}

for class_label, idx in class_labels.items():
    class_folder = os.path.join(dataset_path, class_label)
    for image_file in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_file)
        features = extract_features(image_path)
        X.append(features)
        y.append(idx)
        image_paths.append(image_path)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
    X, y, image_paths, test_size=0.2, stratify=y, random_state=42
)

# training svm model
svm_model = SVC(kernel='linear', C=0.1)
svm_model.fit(X_train, y_train)

# evaluate model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_labels.keys())

print(f"SVM Model Accuracy: {accuracy:.2%}")
print("Classification Report:")
print(report)

# save model
joblib.dump(svm_model, "svm_model_fixed.pkl")
