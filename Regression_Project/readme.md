
# 🎵 ANN Regression Model for Predicting Beats Per Minute (BPM)

This project focuses on **predicting the Beats Per Minute (BPM)** of tracks using a **Deep Learning (Artificial Neural Network) Regression Model** built with TensorFlow/Keras.  
It was developed using data from the [Kaggle Playground Series - Season 5 Episode 9](https://www.kaggle.com/competitions/playground-series-s5e9).

---

## 📘 Project Overview

The goal of this project is to build a **regression model** that predicts the target variable `BeatsPerMinute` from given track-level features.

The end-to-end workflow includes:
- Data Exploration and Visualization  
- Feature Scaling and Preprocessing  
- ANN Model Building using TensorFlow  
- Hyperparameter Tuning using `RandomizedSearchCV`  
- Generating Predictions for Submission  

---

## 🧰 Tech Stack & Libraries Used

**Languages & Frameworks**
- Python 3
- TensorFlow / Keras

**Libraries**
- `pandas`, `numpy` — Data loading and processing  
- `matplotlib`, `seaborn` — Data visualization  
- `scikit-learn` — Preprocessing, train-test split, evaluation  
- `scikeras` — Sklearn wrapper for Keras models  
- `tensorflow` — ANN model implementation  

---

## 📂 Dataset Information

The dataset is provided by Kaggle under the [Playground Series S5E9 competition](https://www.kaggle.com/competitions/playground-series-s5e9).  
It contains a variety of numerical and categorical features describing songs.

| File | Description |
|------|--------------|
| `train.csv` | Training data containing features and the target `BeatsPerMinute` |
| `test.csv` | Test data used for generating predictions |
| `sample_submission.csv` | Sample submission file for Kaggle |

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed using `pandas`, `matplotlib`, and `seaborn` to understand feature distributions and relationships.

- **Boxplots:** Identified outliers  
- **KDE plots:** Observed feature distributions  
- **Correlation heatmap:** Found relationships between features and target

---
### Data Preprocessing

Key preprocessing steps:
- Dropped non-informative id column
- Split the dataset into training and validation sets (85:15 ratio)
- Scaled features using StandardScaler

---
### Model Details

1. Optimizer: Adam
2. Loss Function: Mean Squared Error (MSE)
3. Metric: Root Mean Squared Error (RMSE)
4. EarlyStopping used with patience of 35 epochs to prevent overfitting

---

### Hyperparameter Tuning

To optimize the ANN, RandomizedSearchCV was used with the KerasRegressor wrapper from scikeras.

Parameters tuned:
- Hidden layer units: [64, 128]
- Activation functions: ['relu', 'elu']
- Optimizers: ['adam', 'rmsprop']
- Learning rates: [0.001]
- Batch size: [64]
- Epochs: [20, 30]

---
###
📈 Model Evaluation

- The model was trained using RMSE as a performance metric.
- EarlyStopping ensured that training stopped at the optimal point.
- After hyperparameter tuning, model performance improved significantly.
