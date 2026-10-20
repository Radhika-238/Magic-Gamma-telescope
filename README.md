# MAGIC Gamma Telescope — Machine Learning and Deep Learning Classification

A comparative Machine Learning and Deep Learning project focused on classifying high-energy particle events recorded by the MAGIC (Major Atmospheric Gamma Imaging Cherenkov) Telescope.

The project implements multiple classification algorithms, explores data preprocessing techniques, and evaluates model performance to distinguish gamma-ray events from hadronic background events.

## Project Overview

The MAGIC Telescope detects Cherenkov light produced by particle showers in the Earth's atmosphere.

The goal of this project is to classify observed events into two categories:

- **Gamma (1):** Gamma-ray events
- **Hadron (0):** Background hadronic events

Using numerical features describing the recorded events, different classification algorithms are trained and compared to evaluate their ability to distinguish between the two classes.

## Dataset

**Dataset:** MAGIC Gamma Telescope Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)

| Property | Description |
|---|---|
| Total Samples | 19,020 |
| Input Features | 10 |
| Target Variable | class |
| Problem Type | Binary Classification |
| Feature Type | Numerical |
| Classes | Gamma and Hadron |

### Input Features

The dataset contains ten numerical features:

- `fLength`
- `fWidth`
- `fSize`
- `fConc`
- `fConc1`
- `fAsym`
- `fM3Long`
- `fM3Trans`
- `fAlpha`
- `fDist`

These features describe geometric and statistical characteristics of recorded particle events.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Class Imbalance Handling | Imbalanced-learn |
| Development Environment | Google Colab, Jupyter Notebook |

## Project Workflow

### 1. Data Loading and Exploration

- Loaded the MAGIC Gamma Telescope dataset.
- Assigned descriptive column names.
- Inspected sample records and target labels.
- Converted categorical class labels into binary values.
- Visualized feature distributions for gamma and hadron events.

### 2. Data Preprocessing

- Randomly divided the dataset into training, validation, and testing subsets.
- Used a 60:20:20 split.
- Applied StandardScaler to numerical features.
- Used RandomOverSampler to balance the training dataset.

### 3. Machine Learning Models

Implemented and evaluated the following algorithms:

**K-Nearest Neighbors (KNN)**

- Used KNeighborsClassifier with three neighbors.
- Evaluated predictions using classification metrics.

**Gaussian Naive Bayes**

- Implemented GaussianNB for probabilistic classification.
- Compared its performance with other algorithms.

**Logistic Regression**

- Trained a Logistic Regression classifier.
- Evaluated its ability to distinguish gamma and hadron events.

**Support Vector Machine (SVM)**

- Implemented SVC for nonlinear classification.
- Compared its performance with other classification models.

### 4. Deep Learning Model

Developed a feedforward Artificial Neural Network using TensorFlow/Keras.

**Architecture:**

- Input: 10 numerical features
- Two hidden Dense layers with ReLU activation
- Dropout regularization
- Output layer with sigmoid activation

**Training Configuration:**

- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Evaluation Metric: Accuracy
- Maximum Epoch
