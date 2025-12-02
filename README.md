# Airbnb Superhost Prediction

This project trains a **Logistic Regression model** to predict whether an Airbnb host is a **"super host"** based on listing features. It includes model training, hyperparameter optimization, evaluation, feature selection, and model persistence using `pickle`.

---

## Project Overview

- **Goal:** Predict whether an Airbnb host is a super host (binary classification).  
- **Dataset:** `airbnbData_train.csv`  
  - Preprocessed with one-hot encoding for categorical features, scaling of numerical features, and imputation of missing values.  
- **Model:** Logistic Regression with hyperparameter tuning using `GridSearchCV`  
- **Evaluation Metrics:** Accuracy, Precision-Recall Curve, ROC Curve, AUC  
- **Feature Selection:** SelectKBest to identify top predictive features  
- **Persistent Model:** Saved as `Pickle_airbnb_Logistic_Regression_Model.pkl`

---

## Repository Contents

| File | Description |
|------|-------------|
| `airbnbData_train.csv` | Preprocessed training dataset for Airbnb listings |
| `Pickle_airbnb_Logistic_Regression_Model.pkl` | Serialized Logistic Regression model for predicting super hosts |
| `Pickle_airbnb_Logistic_Regression_Model_info.json` | Metadata about the pickled model |
| `README.md` | This file |

---

## How to Use

1. **Clone the repository:**
```bash
git clone https://github.com/mahashanawaz/airbnb-superhost-prediction.git
````

2. **Load the dataset in Python:**
```bash
import pandas as pd
df = pd.read_csv('airbnbData_train.csv')
````

3. **Load the persisted model:**
```bash
import pickle
model = pickle.load(open('Pickle_airbnb_Logistic_Regression_Model.pkl', 'rb'))
````

4. **Make predictions on new data:**
```bash
X_new = df.drop(columns='host_is_superhost')
predictions = model.predict(X_new)
````
---

## Model Training & Evaluation

### 1. Train/Test Split
- Split the data into 90% training and 10% test set using `train_test_split`.

### 2. Logistic Regression
- Default hyperparameter: `C=1.0`, `max_iter=1000`    
- Optimized hyperparameter `C` found using `GridSearchCV` with 5-fold cross-validation.

### 3. Evaluation Metrics
- **Confusion Matrix**   
- **Precision-Recall Curve**     
- **ROC Curve & AUC**         
- Feature selection using `SelectKBest` to find top predictive features.

### 4. Model Persistence
- The trained Logistic Regression model is saved using `pickle` for future predictions.

---

## Results

- Optimized Logistic Regression achieved **AUC = 0.8187** on the test set.   
- Top predictive features were selected using `SelectKBest`  and further improved model interpretability.

---

## Notes

- Some code cells (like GridSearchCV) may take a few minutes to run.        
- All preprocessing steps are already applied in the dataset; no additional preprocessing is needed before training the model.













