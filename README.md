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
- Maximum Epochs: 100

### 5. Neural Network Hyperparameter Experiments

Experimented with different training configurations:

| Hyperparameter | Values |
|---|---|
| Hidden Layer Neurons | 16, 32, 64 |
| Dropout Rate | 0, 0.2 |
| Learning Rate | 0.01, 0.005, 0.001 |
| Batch Size | 32, 64, 128 |

Compared neural network configurations using validation loss and selected the model with the lowest observed validation loss.

## Model Performance

The following results were obtained from the notebook's classification reports.

| Model | Test Accuracy |
|---|---:|
| K-Nearest Neighbors | 82% |
| Gaussian Naive Bayes | 72% |
| Logistic Regression | 79% |
| Support Vector Machine | 87% |
| Neural Network (TensorFlow/Keras) | 87% |

### Detailed Classification Results

| Model | Gamma Precision | Gamma Recall | Gamma F1 |
|---|---:|---:|---:|
| KNN | 0.86 | 0.86 | 0.86 |
| Gaussian Naive Bayes | 0.74 | 0.89 | 0.81 |
| Logistic Regression | 0.84 | 0.82 | 0.83 |
| SVM | 0.89 | 0.90 | 0.90 |
| Neural Network | 0.87 | 0.95 | 0.91 |

**Key Observation:** The neural network achieved the highest gamma-class recall among the evaluated models, while SVM demonstrated strong overall classification performance.

These results reflect the particular dataset split and experimental setup used in the notebook.

## Neural Network Implementation

The neural network was implemented using TensorFlow's Sequential API.

The model includes:

- Fully connected Dense layers
- ReLU activation functions
- Dropout regularization
- Sigmoid output activation
- Adam optimization
- Binary Cross-Entropy loss

Training and validation loss and accuracy were visualized to examine model learning behavior.

## Project Structure

```text
Magic-Gamma-telescope/
│
├── Magic_Gamma_Telescope.ipynb
├── magic04.data
├── README.md
└── LICENSE
```

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Radhika-238/Magic-Gamma-telescope.git
cd Magic-Gamma-telescope
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow
```

### 3. Run the Notebook

Open `Magic_Gamma_Telescope.ipynb` using Jupyter Notebook or Google Colab.

Ensure that `magic04.data` is accessible in the notebook's working directory.

Execute the cells sequentially to reproduce the data preprocessing, model training, and evaluation workflow.

## Key Learnings

This project provided practical experience in:

- Working with scientific classification datasets.
- Visualizing feature distributions.
- Performing numerical feature scaling.
- Handling imbalanced training data through oversampling.
- Implementing multiple classification algorithms.
- Evaluating models using precision, recall, F1-score, and accuracy.
- Building feedforward neural networks with TensorFlow/Keras.
- Experimenting with neural network hyperparameters.
- Comparing traditional ML and Deep Learning approaches.

## Limitations and Future Improvements

- The notebook fits StandardScaler separately to each dataset split; future implementations should fit the scaler only on training data and reuse it for validation and testing.
- Random splitting may not fully account for the dataset's original event ordering or grouping.
- Neural network experiments involve multiple configurations and can be computationally expensive.
- Future improvements could include stratified splitting, cross-validation, more systematic hyperparameter tuning, and ROC-AUC evaluation.
- Additional comparisons could investigate precision-recall trade-offs and model calibration.

## Author

**Radhika Agarwal**

B.Tech — Computer Science and Engineering

**GitHub:** [Radhika-238](https://github.com/Radhika-238)

---

This project demonstrates the implementation and comparison of traditional Machine Learning classifiers and feedforward neural networks for gamma-ray event classification.
