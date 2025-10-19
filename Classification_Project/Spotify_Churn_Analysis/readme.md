# 🎵 Spotify Churn Prediction – End-to-End ANN Project with Deployment

**Deployed on Streamlit:** [View Live App](https://ann-projects-psjubjnhwmjbsxnhut6dde.streamlit.app/)

**Python | TensorFlow/Keras | ANN | StandardScaler | ColumnTransformer**

---

## 📑 Table of Contents
- Project Overview
- Dataset
- Data Exploration & Visualization
- Data Preprocessing
- Model Building & Training
- Hyperparameter Tuning
- Final Model Performance
- Predictions
- Key Takeaways
- Technologies & Libraries Used

---

## Project Overview
This project predicts whether a Spotify user will **churn** based on their usage patterns, session behavior, and subscription metrics.  
The goal is to help Spotify identify users at risk of churn and take retention actions to improve user engagement.

---

## Dataset
- **Source:** [Kaggle Spotify Dataset](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis/data)
- **Rows:** 8,000  
- **Columns:** 12  
  - `user_id` (int)  
  - `gender` (object)  
  - `age` (int)  
  - `country` (object)  
  - `subscription_type` (object)  
  - `listening_time` (int)  
  - `songs_played_per_day` (int)  
  - `skip_rate` (float)  
  - `device_type` (object)  
  - `ads_listened_per_week` (int)  
  - `offline_listening` (int)  
  - `is_churned` (target, int: 0 = No, 1 = Yes)

---

## Data Exploration & Visualization
- Checked for missing/null values → none found.  
- Explored numerical distributions with **boxplots**.  
- Explored categorical features with **countplots**.  
- Visualized target variable with a **pie chart** to check class balance.  

---

## Data Preprocessing
- Dropped `user_id` as irrelevant.  
- Split dataset into **train (75%)** and **test (25%)** sets.  
- Used `StandardScaler` for numerical features and `OneHotEncoder` for categorical features.  
- Implemented a **ColumnTransformer** to combine scaling and encoding.  
- Saved preprocessing pipeline as `processing.pkl` for deployment consistency.  

---

## Model Building & Training
- Built an **Artificial Neural Network (ANN)** using Keras:
  - Input layer: 128 neurons, activation = 'elu'  
  - Hidden layers: 64, 32 neurons, activation = 'elu'  
  - Output layer: 1 neuron, activation = 'sigmoid'  
- Compiled with **binary_crossentropy** loss and **Adam optimizer**.  
- Trained for 100 epochs with **EarlyStopping** (patience=20) monitoring validation loss.  
- Achieved validation accuracy ~0.748.  

---

## Hyperparameter Tuning
- Wrapped ANN in `KerasClassifier` for use with `GridSearchCV`.  
- Tuned:
  - Number of neurons in each hidden layer  
  - Activation functions (`relu`, `elu`)  
  - Optimizers (`adam`, `rmsprop`)  
  - Learning rate, batch size, epochs  

**Best parameters found:**
```python
{
    'n_hidden1': 64,
    'n_hidden2': 32,
    'n_hidden3': 16,
    'activation': 'elu',
    'optimizer': 'rmsprop',
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 20
}
```
---
### Final Model Performance

| Metric    | ANN Model |
| --------- | --------- |
| Accuracy  | 0.748     |
| Precision | 0.75      |
| Recall    | 0.74      |
| F1-Score  | 0.745     |

- The trained ANN outperforms simple models due to capturing complex user behavior patterns.
  
---

### Predictions

- Users input features such as:
  - Gender, Age, Country
  - Subscription Type
  - Listening Time, Songs Played per Day
  - Skip Rate, Device Type
  - Ads Listened, Offline Listening

- The trained ANN predicts the likelihood of churn and outputs a probability.

---
### Key Takeaways 

- ANN effectively captures non-linear relationships in user behavior.
- Preprocessing with scaling and encoding ensures consistency in deployment.
- Hyperparameter tuning significantly improved model performance.
- Real-time predictions via Streamlit deployment allow business teams to identify at-risk users.

---

### Technologies & Liraries Used

- Python 3.10
- `pandas`, `numpy`
- `scikit-learn` (StandardScaler, OneHotEncoder, ColumnTransformer, train_test_split)
- `tensorflow` / `keras` (Sequential, Dense, EarlyStopping)
- `scikeras.wrappers.KerasClassifier`, `GridSearchCV`
- `matplotlib`, `seaborn`
- `pickle`
- `streamlit`

  
