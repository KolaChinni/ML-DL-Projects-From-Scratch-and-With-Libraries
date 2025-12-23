# 👕 Fashion-MNIST Classification

This project implements **three different approaches** to solve a **multiclass image classification problem** using the **Fashion-MNIST dataset**.

The goal of this project is to **compare from-scratch neural network implementations with framework-based deep learning models**, progressing from:
- Pure **NumPy implementations**
- To **fully connected neural networks** using TensorFlow/Keras
- To **Convolutional Neural Networks (CNNs)** for improved performance

---

## 📊 Dataset Information

- **Dataset**: Fashion-MNIST
- **Classes**: 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Training Samples**: 60,000
- **Test Samples**: 10,000
- **Image Size**: 28 × 28 (grayscale)

---

## 🔹 Code 1: Neural Network from Scratch (NumPy + Adam)
# 👕 Fashion-MNIST Classification – From Scratch to Deep Learning

This project implements **three different approaches** to solve a **multiclass image classification problem** using the **Fashion-MNIST dataset**.

The objective is to compare:
- **From-scratch neural network implementations (NumPy)**
- **Framework-based fully connected neural networks**
- **Convolutional Neural Networks (CNNs)**

This progression highlights how model complexity and libraries impact performance.

---

## 📊 Dataset Information

- **Dataset**: Fashion-MNIST  
- **Classes**: 10  
- **Training Samples**: 60,000  
- **Test Samples**: 10,000  
- **Image Size**: 28 × 28 (grayscale)

---

## 🔹 Code 1: Neural Network from Scratch (NumPy + Adam)

### 📌 Description
A **fully connected neural network implemented entirely from scratch using NumPy**, including:
- Forward propagation
- Backpropagation
- Categorical cross-entropy loss
- **Adam optimizer (from scratch)**
- Mini-batch training

TensorFlow is used **only to load the dataset**.

---

### 🧠 Architecture

| Layer | Units | Activation |
|-----|------|-----------|
| Input | 784 | — |
| Hidden | 128 | ReLU |
| Output | 10 | Softmax |

---

### ⚙️ Training Details
- Optimizer: Adam (from scratch)  
- Loss: Categorical Cross-Entropy  
- Batch Size: 64  
- Learning Rate: 0.001  
- Early Stopping: Stops around **95–96% training accuracy**

---

### 📊 Performance
- **Training Accuracy: ~95–96%**
- **Test Accuracy: ~88–90%**

---

## 🔹 Code 2: Fully Connected Neural Network (TensorFlow / Keras)

### 📌 Description
A **framework-based feedforward neural network** implemented using **TensorFlow/Keras**, showing faster convergence and simpler implementation compared to from-scratch code.

---

### 🧠 Architecture

| Layer | Details |
|-----|--------|
| Input | Flatten (28×28 → 784) |
| Hidden | Dense (128 units, ReLU) |
| Output | Dense (10 units, Softmax) |

---

### ⚙️ Training Setup
- Optimizer: Adam  
- Loss Function: Sparse Categorical Cross-Entropy  
- Epochs: Up to 50  
- Callback: Early stopping at **95% training accuracy**

---

### 📊 Performance
- **Training Accuracy: ~95–96%**
- **Test Accuracy: ~86–89%**

---

## 🔹 Code 3: Convolutional Neural Network (CNN) – TensorFlow / Keras

### 📌 Description
A **Convolutional Neural Network (CNN)** that leverages spatial features in images, achieving the **best generalization performance**.

---

### 🧠 Architecture

| Layer | Description |
|-----|------------|
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Output | 10 units, Softmax |

---

### ⚙️ Training Setup
- Optimizer: Adam  
- Loss Function: Sparse Categorical Cross-Entropy  
- Epochs: Up to 100  
- Callback: Early stopping at **99% training accuracy**

---

### 📊 Performance
- **Training Accuracy: ~99%**
- **Test Accuracy: ~90–95%**

---

## 📈 Performance Comparison

| Model | Implementation | Training Accuracy | Test Accuracy |
|-----|---------------|------------------|--------------|
| Fully Connected NN | NumPy (from scratch) | ~95% | ~86–88% |
| Fully Connected NN | TensorFlow / Keras | ~95–96% | ~86–89% |
| CNN | TensorFlow / Keras | ~99% | ~90–95% |

---

## 🧠 Key Learning Outcomes

- Implementing neural networks and Adam optimizer from scratch
- Understanding training vs generalization gap
- Benefits of deep learning frameworks
- Importance of CNNs for image classification
- Comparing architectures on the same dataset

---

## 🛠 Technologies Used

- Python  
- NumPy  
- TensorFlow / Keras  
- scikit-learn  
- Fashion-MNIST dataset  

---

## 🏁 Conclusion

This project demonstrates a **complete progression from low-level neural network implementations to advanced CNN-based models**, clearly showing how **model design and tools affect training and test performance**.

---

⭐ If you find this project useful, feel free to star the repository!

### 📌 Description
A **fully connected neural network implemented entirely from scratch using NumPy**, with:
- Manual forward propagation
- Backpropagation
- Categorical cross-entropy loss
- **Adam optimizer implemented manually**
- Mini-batch training

TensorFlow is used **only for loading the dataset**.

---

### 🧠 Architecture
| Layer | Units | Activation |
|------|------|-----------|
| Input | 784 | — |
| Hidden | 128 | ReLU |
| Output | 10 | Softmax |

---

### ⚙️ Training Details
- Optimizer: Adam (from scratch)
- Loss: Categorical Cross-Entropy
- Batch Size: 64
- Learning Rate: 0.001
- Early Stopping: Stops at ~95–96% training accuracy

---

### ✅ Performance
- **Test Accuracy: ~88–92%** (varies per run)

---

## 🔹 Code 2: Fully Connected Neural Network (TensorFlow / Keras)

### 📌 Description
A **framework-based feedforward neural network** implemented using **TensorFlow/Keras**, demonstrating how deep learning libraries simplify model training and evaluation.

---

### 🧠 Architecture
| Layer | Details |
|------|--------|
| Input | Flatten (28×28 → 784) |
| Hidden | Dense (128 units, ReLU) |
| Output | Dense (10 units, Softmax) |

---

### ⚙️ Training Setup
- Optimizer: Adam
- Loss Function: Sparse Categorical Cross-Entropy
- Epochs: Up to 50
- Callback: Early stopping at **95% accuracy**

---

### 📊 Evaluation
- Classification Report
- Confusion Matrix
- Accuracy Score

---

### ✅ Performance
- **Test Accuracy: ~95–96%**

---

## 🔹 Code 3: Convolutional Neural Network (CNN) – TensorFlow / Keras

### 📌 Description
A **Convolutional Neural Network (CNN)** designed to leverage the spatial structure of image data, achieving the **highest accuracy** among the three approaches.

---

### 🧠 Architecture
| Layer | Description |
|------|------------|
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Output | 10 units, Softmax |

---

### ⚙️ Training Setup
- Optimizer: Adam
- Loss Function: Sparse Categorical Cross-Entropy
- Epochs: Up to 100
- Callback: Early stopping at **99% accuracy**

---

### 📊 Evaluation
- Classification Report
- Confusion Matrix
- Accuracy Score

---

### ✅ Performance
- **Test Accuracy: ~98–99%**

---

## 📈 Performance Comparison

| Model | Implementation | Test Accuracy |
|-----|---------------|--------------|
| Fully Connected NN | NumPy (from scratch) | ~88–92% |
| Fully Connected NN | TensorFlow / Keras | ~95–96% |
| CNN | TensorFlow / Keras | ~98–99% |

---

## 🧠 Key Learning Outcomes

- Understanding neural networks at a mathematical level
- Implementing Adam optimizer from scratch
- Mini-batch training and softmax classification
- Benefits of deep learning frameworks
- Importance of CNNs for image data
- Comparing performance across architectures

---

## 🛠 Technologies Used

- Python
- NumPy
- TensorFlow / Keras
- scikit-learn
- Fashion-MNIST dataset

---

## 🏁 Conclusion

This project demonstrates a **complete progression from foundational neural networks to advanced deep learning models**, highlighting the strengths of both **from-scratch implementations** and **modern deep learning frameworks**.

---

⭐ If you find this project useful, feel free to star the repository!
