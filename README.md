

# 🧠 Image Classification with Scikit-learn

This project demonstrates a simple image classification pipeline using **color histogram features** and a **Support Vector Machine (SVM)** classifier with `scikit-learn`. It classifies images into categories like **cats**, **dogs**, and **wild animals** using only color information.

---

## 📁 Folder Structure

Make sure your dataset is structured like this:
```
dataset/
└── train/
|    ├── cat/
|    │   ├── image1.jpg
|    │   ├── image2.jpg
|    │   └── …
|    ├── dog/
|    │   ├── image1.jpg
|    │   ├── image2.jpg
|    │   └── …
|    └── wild/
|        ├── image1.jpg
|        ├── image2.jpg
|        └── …
└── val/
|    ├── cat/
|    │   ├── image1.jpg
|    │   ├── image2.jpg
|    │   └── …
|    ├── dog/
|    │   ├── image1.jpg
|    │   ├── image2.jpg
|    │   └── …
|    └── wild/
|        ├── image1.jpg
|        ├── image2.jpg
|        └── …
```
Each subfolder name will be used as a class label.

---

## 🚀 Features

- 📸 Feature extraction using 3D color histograms (RGB)
- 🧠 Model training using Support Vector Machine (SVM)
- 📊 Evaluation using `classification_report`
- 🔁 Train/Test splitting
- 🔍 Label encoding and normalization
- 🔎 Visual predictions preview in console

---

## ⚙️ Requirements

Install the following Python packages before running:

```bash
pip install numpy matplotlib scikit-learn opencv-python pillow

```
⸻

# 📜 Usage

Run the Python script:
```
python image_classifier.py
```
It will:
1.	Load images from dataset/train or val/
2.	Extract features using color histograms
3.	Train a Support Vector Machine model
4.	Evaluate accuracy
5.	Print predicted vs actual labels for a few samples

⸻

# 🧪 Sample Output

Loading dataset...
Total samples loaded: 120
```
Classification Report:
              precision    recall  f1-score   support

         cat       0.86      0.83      0.84        30
         dog       0.89      0.92      0.90        25
        wild       0.85      0.88      0.86        25

    accuracy                           0.87        80
   macro avg       0.87      0.87      0.87        80
weighted avg       0.87      0.87      0.87        80

Sample 1: True = dog, Predicted = dog  
Sample 2: True = cat, Predicted = cat  
Sample 3: True = wild, Predicted = wild  
Sample 4: True = cat, Predicted = dog  
Sample 5: True = wild, Predicted = wild
```

⸻

# 📝 How it Works
1.	Color Histogram Extraction
Each image is resized and converted into a 3D RGB histogram (8 bins per channel).
2.	Feature Vector Creation
The histogram is normalized and flattened into a 512-length feature vector.
3.	Training
Uses sklearn.svm.SVC with a linear kernel to learn the patterns.
4.	Testing & Evaluation
Split dataset into 80% training, 20% testing and evaluate with classification_report.

⸻

## 📌 To-Do
- Add GUI using Tkinter or PyQt
- Export trained model with joblib or pickle
- Add support for command-line prediction of single image
- Integrate TensorFlow/Keras for CNN-based comparison

⸻

## 👨‍💻 Author

Abdullah Nazmus-Sakib
- 📫 Email : shakibrybmn@gmail.com
- GitHub : AbdullahRFA

⸻
# Dataset Downloaded
From : https://gts.ai/dataset-download/animal-faces-dataset-ai-data-collection/?utm_source=chatgpt.com#wpcf7-f47097-o1
---
📜 License

This project is open source and available under the MIT License.

---

