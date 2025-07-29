# Import essential libraries
import os                      # For file system operations
import cv2                     # OpenCV for image loading
import numpy as np             # For numerical operations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from PIL import Image          # For image verification

# Set the path to your dataset folder (update if needed)
dataset_path = "dataset/val"

# Function to extract color histogram from an image
def extract_color_histogram(image_path):
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert to RGB because OpenCV loads in BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate 3D histogram in RGB space with 8 bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])

        # Normalize the histogram and flatten it to 1D array
        hist = cv2.normalize(hist, hist).flatten()

        return hist
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Function to load dataset and prepare X (features) and y (labels)
def load_dataset(path):
    X = []  # Feature list
    y = []  # Label list

    # Loop through each label/class folder (cat, dog, wild)
    for label in os.listdir(path):
        class_dir = os.path.join(path, label)

        if not os.path.isdir(class_dir):
            continue

        # Loop through each image in the folder
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, file)

                # Extract histogram features
                features = extract_color_histogram(img_path)

                # If feature extraction was successful
                if features is not None:
                    X.append(features)
                    y.append(label)

    return np.array(X), np.array(y)

# Load the dataset
print("Loading dataset...")
X, y = load_dataset(dataset_path)
print(f"Total samples loaded: {len(X)}")

# Encode labels into numbers (e.g., cat=0, dog=1, wild=2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test set (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the classifier (Support Vector Machine)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Show classification results
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Optional: Visualize a few predictions
def visualize_predictions(X_test, y_test, y_pred, le, num=5):
    for i in range(num):
        # Get the image histogram and decode label
        hist = X_test[i]
        true_label = le.inverse_transform([y_test[i]])[0]
        pred_label = le.inverse_transform([y_pred[i]])[0]

        print(f"Sample {i+1}: True = {true_label}, Predicted = {pred_label}")

# Show predictions for 5 test samples
visualize_predictions(X_test, y_test, y_pred, le, num=5)