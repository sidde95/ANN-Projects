# ANN Projects

Welcome to my **Artificial Neural Network (ANN)** repository!  
This repository features end-to-end Deep Learning projects built with **TensorFlow** and **Keras**, demonstrating how ANNs can solve both **classification** and **regression** problems in real-world data scenarios.

Each project includes full **data preprocessing, model building, tuning, and Streamlit deployment**, making it both educational and production-ready.

---

## Projects Overview

###  [1. Spotify Churn Prediction – Classification](Classification_Project/Spotify_Churn_Analysis)

An end-to-end **Artificial Neural Network (ANN)** project that predicts whether a Spotify user is likely to churn based on their usage patterns and listening behavior.  
The system helps identify at-risk users, enabling proactive retention strategies.

**Framework:** TensorFlow / Keras  
**Model Type:** Binary Classification  
**Deployment:** Streamlit (Live App)  

**Key Features**
- Built an ANN with multiple hidden layers using ELU activations.  
- Implemented `StandardScaler` and `ColumnTransformer` for preprocessing.  
- Tuned model hyperparameters via `GridSearchCV` (KerasClassifier wrapper).  
- Achieved **~74.8% validation accuracy** after tuning.  
- Real-time predictions through a Streamlit interface.  

**Technologies Used**  
Python | TensorFlow/Keras | scikit-learn | pandas | numpy | seaborn | matplotlib | Streamlit  

**Live App:** [View on Streamlit](#)

---

### [2. Beats Per Minute (BPM) Prediction – Regression](Regression_Project/Predicting%20the%20Beats-per-Minute%20of%20Songs)

A Deep Learning regression model that predicts the **Beats Per Minute (BPM)** of tracks using an **ANN built with TensorFlow/Keras**, developed for the [Kaggle Playground Series - S5E9](https://www.kaggle.com/competitions/playground-series-s5e9).

** Framework:** TensorFlow / Keras  
** Model Type:** Regression  
** Tuning:** RandomizedSearchCV with KerasRegressor  

**Key Features**
- Full data exploration and preprocessing using pandas, matplotlib, seaborn.  
- Scaled features using `StandardScaler` for stable ANN convergence.  
- Applied `EarlyStopping` (patience = 35) to prevent overfitting.  
- Tuned model parameters such as activations, optimizers, and epochs.  
- Optimized model performance using **RMSE** as the evaluation metric.  

**Technologies Used**  
Python | TensorFlow/Keras | scikit-learn | pandas | numpy | seaborn | matplotlib  

**Competition Source:** [Kaggle Playground S5E9](https://www.kaggle.com/competitions/playground-series-s5e9)

---

## Common Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3.10 |
| **Deep Learning** | TensorFlow, Keras |
| **Data Handling** | pandas, numpy |
| **Preprocessing** | StandardScaler, OneHotEncoder, ColumnTransformer |
| **Model Tuning** | scikeras (KerasClassifier/KerasRegressor), GridSearchCV, RandomizedSearchCV |
| **Visualization** | matplotlib, seaborn |
| **Deployment** | Streamlit |
| **Utilities** | pickle, dotenv |

---

## Key Learnings & Highlights
- Designed **ANN architectures** for both regression and classification.  
- Applied **systematic hyperparameter optimization** for deep learning models.  
- Leveraged **modern deployment practices** for live AI apps using Streamlit.  
- Developed **scalable preprocessing pipelines** with scikit-learn transformers.  
- Delivered **interpretable results** for business and creative applications alike.  



