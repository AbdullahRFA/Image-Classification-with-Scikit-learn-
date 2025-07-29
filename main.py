import os
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# STEP 1: Feature extraction function
def extract_color_histogram(image_path):
    try:
        image = Image.open(image_path).convert('RGB').resize((64, 64))
        image_np = np.array(image)
        hist = []

        for channel in range(3):  # R, G, B channels
            hist_channel, _ = np.histogram(
                image_np[:, :, channel], bins=256, range=(0, 256)
            )
            hist.extend(hist_channel)

        hist = np.array(hist)
        hist = hist / np.sum(hist)  # Normalize histogram
        return hist
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# STEP 2: Load dataset
def load_dataset(dataset_dir):
    X, y = [], []
    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, file)
                features = extract_color_histogram(img_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

# STEP 3: Train and evaluate model
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    return model

# STEP 4: Predict single image
def predict_image(model, image_path):
    features = extract_color_histogram(image_path)
    if features is None:
        return "Invalid image"
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# STEP 5: Visual test
def show_prediction(model, image_path):
    prediction = predict_image(model, image_path)
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f"Predicted: {prediction}")
    plt.axis('off')
    plt.show()

# Main
if __name__ == "__main__":
    dataset_path = "dataset"
    print("Loading dataset...")
    X, y = load_dataset(dataset_path)

    print(f"Dataset loaded: {len(X)} samples.")
    model = train_and_evaluate(X, y)

    # Try predicting on a new image
    test_image = input("Enter path to image to test (e.g., dataset/dogs/dog1.jpg): ")
    if os.path.exists(test_image):
        show_prediction(model, test_image)
    else:
        print("Invalid file path!")