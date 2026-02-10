# Customer Churn Prediction with Artificial Neural Networks

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-0078D4?style=for-the-badge&logo=databricks&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Preprocessing](https://img.shields.io/badge/Preprocessing-4CAF50?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Data Analysis](https://img.shields.io/badge/Data%20Analysis-E91E63?style=for-the-badge)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)

<br/>

*An end-to-end deep learning project for predicting customer churn in the telecom industry*

</div>

---

## Problem Statement

Customer churn is one of the most critical challenges faced by telecom companies. Acquiring a new customer costs **5-7x more** than retaining an existing one. This project builds a production-ready ANN model that identifies high-risk customers so the business can launch **targeted retention campaigns** before they leave.

Using the **IBM Telco Customer Churn** dataset (7,043 customers, 33 features), we develop a binary classifier that predicts whether a customer will churn or not.

---

## Highlights

- **Baseline vs Regularized** - demonstrates overfitting, then fixes it with Dropout + L2
- **Hyperparameter Tuning** - systematic comparison of 4 architecture configurations
- **Business-Focused Evaluation** - confusion matrix analysis with cost interpretation
- **Interview-Ready Code** - clean, well-commented, section-by-section explanations
- **Deployment-Ready** - model, scaler, and encoders saved as reusable artifacts

---

## Dataset

**Source**: [IBM Telco Customer Churn](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) (Excel format)

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Male / Female |
| Senior Citizen | Categorical | Yes / No |
| Partner | Categorical | Has partner |
| Dependents | Categorical | Has dependents |
| Tenure Months | Numerical | Months with company |
| Phone Service | Categorical | Has phone service |
| Multiple Lines | Categorical | Has multiple lines |
| Internet Service | Categorical | DSL / Fiber optic / No |
| Online Security | Categorical | Has online security |
| Online Backup | Categorical | Has online backup |
| Device Protection | Categorical | Has device protection |
| Tech Support | Categorical | Has tech support |
| Streaming TV | Categorical | Streams TV |
| Streaming Movies | Categorical | Streams movies |
| Contract | Categorical | Month-to-month / One year / Two year |
| Paperless Billing | Categorical | Uses paperless billing |
| Payment Method | Categorical | Payment type |
| Monthly Charges | Numerical | Monthly bill amount |
| Total Charges | Numerical | Total amount billed |
| **Churn Value** | **Target** | **0 = No, 1 = Yes** |

> **Dropped 13 columns**: CustomerID, Count, Country, State, City, Zip Code, Lat Long, Latitude, Longitude, Churn Label, Churn Score, CLTV, Churn Reason (irrelevant or cause data leakage)

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow, Keras |
| ML Utilities | scikit-learn (StandardScaler, LabelEncoder, metrics) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Serialization | Pickle, HDF5 |

---

## Project Structure

```
Churn prediction with ANN/
|
|-- churn_prediction_ann.ipynb      # Main notebook (all sections)
|-- Telco_customer_churn.xlsx       # IBM Telco dataset
|-- README.md                       # Project documentation
|
|-- churn_prediction_model.h5       # Final saved model (generated)
|-- best_model.h5                   # Best checkpoint (generated)
|-- scaler.pkl                      # StandardScaler (generated)
|-- label_encoders.pkl              # LabelEncoders (generated)
|-- feature_names.pkl               # Feature list (generated)
```

---

## Workflow

```
Data Loading --> EDA --> Cleaning --> Label Encoding --> Train/Test Split
     |                                                        |
     v                                                        v
Churn Distribution                                     StandardScaler
     |                                                        |
     v                                                        v
Baseline ANN (no regularization) -----> Overfitting Demo
     |
     v
Regularized ANN (Dropout + L2) -------> Reduced Overfitting
     |
     v
Hyperparameter Tuning (4 configs) ----> Best Config Selection
     |
     v
Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
     |
     v
Business Interpretation + Model Saving
```

---

## Model Architecture

### Baseline Model (demonstrates overfitting)

```
Input --> Dense(64, ReLU) --> Dense(32, ReLU) --> Dense(1, Sigmoid)
```

### Regularized Model (production)

```
Input --> Dense(64, ReLU, L2=0.001) --> Dropout(0.3)
      --> Dense(32, ReLU, L2=0.001) --> Dropout(0.3)
      --> Dense(1, Sigmoid)
```

### Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Hidden Activation | **ReLU** | No vanishing gradient, computationally efficient: `max(0, x)` |
| Output Activation | **Sigmoid** | Outputs probability between 0 and 1 for binary classification |
| Loss Function | **Binary Crossentropy** | Standard loss for binary classification, penalizes confident wrong predictions |
| Optimizer | **Adam** | Adaptive learning rates per parameter, combines Momentum + RMSProp |
| Regularization | **Dropout + L2** | Dropout prevents neuron co-adaptation, L2 penalizes large weights |
| Early Stopping | **patience=20** | Stops training when validation loss plateaus, restores best weights |

---

## Hyperparameter Tuning

| Config | Layers | Dropout | L2 | Learning Rate | Batch Size |
|--------|--------|---------|----|---------------|------------|
| A | [128, 64] | 0.3 | 0.001 | 0.001 | 32 |
| B | [64, 32] | 0.4 | 0.01 | 0.001 | 64 |
| C | [128, 64, 32] | 0.2 | 0.001 | 0.0005 | 32 |
| D | [256, 128, 64] | 0.3 | 0.001 | 0.001 | 64 |

Best configuration is selected based on **lowest validation loss**.

---

## Evaluation Metrics

The model is evaluated using multiple metrics to get a complete picture:

- **Accuracy** - overall correctness
- **Precision** - of predicted churners, how many actually churned
- **Recall** - of actual churners, how many did the model catch
- **F1-Score** - harmonic mean of precision and recall
- **ROC-AUC** - model's ability to distinguish between classes
- **Confusion Matrix** - detailed breakdown of TP, TN, FP, FN

### Business Impact of Errors

| Error Type | Meaning | Business Cost |
|------------|---------|---------------|
| **False Positive** | Predicted churn but customer stayed | Moderate - wasted retention budget (discounts, offers) |
| **False Negative** | Predicted stay but customer actually churned | **HIGH** - lost revenue and customer lifetime value |

> In telecom churn prediction, **False Negatives are far more costly**. Consider lowering the classification threshold below 0.5 to improve recall and catch more at-risk customers.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/S07singh/Churn-prediction-with-ANN.git
cd Churn-prediction-with-ANN
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow openpyxl
```

### 3. Run the Notebook

1. Open `churn_prediction_ann.ipynb` in Jupyter Notebook or VS Code
2. Run all cells from top to bottom
3. Model and artifacts will be saved automatically

---

## Model Deployment (Inference)

```python
import tensorflow as tf
import pickle
import numpy as np

# Load saved artifacts
model = tf.keras.models.load_model('churn_prediction_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Prepare new customer data (encode + scale)
scaled_data = scaler.transform(new_customer_data)

# Predict
churn_probability = model.predict(scaled_data)
prediction = (churn_probability > 0.5).astype(int)
```

---

## Key Takeaways (Interview Ready)

1. **Data Leakage Prevention** - scaler is fitted only on training data, leakage columns are dropped
2. **Label Encoding over OHE** - prevents feature explosion, works well with neural networks
3. **Overfitting Demonstration** - baseline model shows train-val gap, regularization closes it
4. **Dropout** - randomly disables neurons during training to force independent feature learning
5. **L2 Regularization** - adds weight penalty to loss function, keeps weights small
6. **EarlyStopping** - monitors val_loss, stops training when improvement plateaus
7. **Stratified Split** - preserves class ratio in train/test to handle imbalanced data
8. **Business Context** - FN is costlier than FP in churn, recall matters more than precision

---

## License

This project is licensed under the **Apache 2.0 License**.

---

<div align="center">

**Built with TensorFlow and Keras**

If you found this project helpful, give it a star!

</div>
