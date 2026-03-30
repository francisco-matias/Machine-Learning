# Machine-Learning

This project explores machine learning techniques applied to both regression and computer vision tasks, focusing on model performance, robustness and real-world data challenges such as noise and class imbalance.

---

## 📄 Report

You can find the full project report here:

👉 [Project Report](https://raw.githubusercontent.com/francisco-matias/Machine-Learning/39cb0e5a9ae8688d58fc4cfe2ea2cda465d4b2a9/ML_report.pdf)

---

## 📌 Overview

The project is divided into two main components:

- **Regression with Synthetic Data**
- **Medical Image Classification (MEDMNIST)**

---

## 🔹 Regression Tasks

### 1. Linear Regression Models
- Implemented and compared:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
- Applied **k-fold cross-validation (k=5)**
- Metrics:
  - Mean Squared Error (MSE)
  - \( R^2 \)

### 2. Dual-Model Regression with Outlier Separation
- Developed a **recursive residual-based algorithm** to:
  - Detect and separate outliers
  - Fit independent models for distinct data distributions
- Explored:
  - K-Means
  - Gaussian Mixture Models
  - RANSAC
- Selected based on best performance

---

## 🔹 Image Classification Tasks

### Dataset
- MEDMNIST medical image datasets
- RGB images (28×28)

### Tasks
- Binary classification (Melanoma vs Nevus)
- Multi-class classification (6 classes)

---

## 🧠 CNN Architecture

- 3 convolutional layers (3×3 kernels, ReLU)
- MaxPooling layers
- Fully connected layers (1024 → 256 → 56)
- Dropout regularization
- Softmax output layer

### Training
- Optimizer: Adam
- Learning rate: 0.0002
- Loss: Cross-entropy
- Early stopping

---

## ⚖️ Handling Data Imbalance

- SMOTE
- Data augmentation:
  - Rotation
  - Flipping
  - Brightness variation
- Class weighting

---

## 📊 Results

- Strong performance in regression and classification tasks
- Effective handling of highly imbalanced datasets
- Top-tier performance in classification benchmarks

---

## 🛠️ Technologies

- Python
- NumPy / Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

---

## 📈 Skills Demonstrated

- Machine learning model development
- Cross-validation and model evaluation
- Outlier detection
- Deep learning (CNNs)
- Image processing
- Imbalanced data handling
