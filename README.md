Support Vector Machine (SVM) Project
Overview

This project demonstrates the implementation of a Support Vector Machine (SVM) classifier in two different ways:

With Library – Using the scikit-learn library for efficient and optimized SVM training.
Without Library – Implementing the core SVM algorithm manually using Python and NumPy.

The project helps in understanding both the practical usage of SVM and the mathematical concepts behind it.

1. SVM Using Library
Description

This implementation uses the scikit-learn machine learning library to train and test an SVM model.

Features
Fast and optimized implementation
Easy model training and prediction
Supports multiple kernels (Linear, Polynomial, RBF)
High accuracy on classification tasks
Technologies Used
Python
NumPy
Pandas
Matplotlib
Scikit-learn
Installation

Install the required libraries using:

pip install numpy pandas matplotlib scikit-learn
Dataset

You can use any classification dataset such as:

Iris Dataset
Breast Cancer Dataset
Custom CSV Dataset

Example datasets are available in scikit-learn.datasets.

Example Code
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create SVM model
model = SVC(kernel='linear')


# Train model
model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)


# Accuracy
print("Accuracy:", accuracy_score(y_test, predictions))
Running the Project
python svm_with_library.py
Advantages
Easy to implement
Highly optimized
Suitable for large datasets
Minimal code required
Limitations
Less understanding of internal working
Dependency on external libraries
2. SVM Without Library
Description

This implementation builds an SVM classifier manually using mathematical concepts and NumPy operations.

Features
Understands the internal working of SVM
Manual implementation of hyperplane optimization
Educational and beginner-friendly
Technologies Used
Python
NumPy
Matplotlib
Installation

Install the required libraries:

pip install numpy matplotlib
Mathematical Concept

SVM tries to find the best hyperplane that separates classes with the maximum margin.

Main equation:

w⋅x+b=0



Where:

w = weight vector
x = input features
b = bias
Example Code
import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None


    def fit(self, X, y):
        n_samples, n_features = X.shape


        y_ = np.where(y <= 0, -1, 1)


        self.w = np.zeros(n_features)
        self.b = 0


        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1


                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
Running the Project
python svm_without_library.py
Advantages
Better understanding of SVM concepts
No dependency on machine learning libraries
Helps in learning optimization techniques
Limitations
Slower compared to optimized libraries
More complex implementation
Limited scalability
Comparison Between With and Without Library
Feature	With Library	Without Library
Ease of Use	Very Easy	Moderate
Performance	High	Medium
Learning Purpose	Low	High
Code Complexity	Low	High
Optimization	Built-in	Manual
Scalability	Excellent	Limited
Project Structure
SVM-Project/
│
├── svm_with_library.py
├── svm_without_library.py
├── dataset.csv
└── README.md
Applications of SVM
Image Classification
Face Detection
Spam Detection
Text Classification
Bioinformatics
Handwriting Recognition
Future Improvements
Add kernel trick implementation
Support multiclass classification
Improve optimization algorithm
Add visualization for decision boundary
Conclusion

This project provides both a practical and theoretical understanding of Support Vector Machines. Using libraries helps in building models quickly, while manual implementation helps in understanding the mathematics and optimization behind SVM.

Author

Developed for learning and educational purposes.
