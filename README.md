# Customer Churn Prediction with Artificial Neural Networks

## Project Overview

End-to-end deep learning project to predict customer churn in the telecom industry using the IBM Telco Customer Churn dataset (7043 customers, 33 columns). Built with TensorFlow/Keras.

## Business Problem

Customer churn directly impacts revenue. This model identifies high-risk customers so the business can run targeted retention campaigns before they leave.

## Dataset

**Source**: IBM Telco Customer Churn (Excel format)

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

**Dropped Columns**: CustomerID, Count, Country, State, City, Zip Code, Lat Long, Latitude, Longitude, Churn Label, Churn Score, CLTV, Churn Reason (irrelevant or data leakage)

## Tech Stack

- Python 3.8+
- TensorFlow / Keras
- scikit-learn
- pandas, NumPy
- Matplotlib, Seaborn

## Project Structure

```
Churn prediction with ANN/
  churn_prediction_ann.ipynb    # Complete notebook
  Telco_customer_churn.xlsx     # Dataset
  README.md                     # This file
  churn_prediction_model.h5     # Saved model (after running)
  best_model.h5                 # Best checkpoint (after running)
  scaler.pkl                    # StandardScaler (after running)
  label_encoders.pkl            # LabelEncoders (after running)
  feature_names.pkl             # Feature list (after running)
```

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow openpyxl
```

## How to Run

1. Place `Telco_customer_churn.xlsx` in the project folder
2. Open `churn_prediction_ann.ipynb` in Jupyter Notebook
3. Run all cells top to bottom

## Approach

### Data Preprocessing
- Dropped 13 irrelevant/leakage columns
- Converted Total Charges from object to float, filled missing with median
- **Label Encoding** for all categorical features (avoids feature explosion from One-Hot Encoding)
- 80/20 train-test split with stratification
- StandardScaler fitted on training data only (no data leakage)

### Model Architecture

**Baseline Model** (demonstrates overfitting):
```
Input -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Sigmoid)
```

**Regularized Model** (production):
```
Input -> Dense(64, ReLU, L2) -> Dropout(0.3) -> Dense(32, ReLU, L2) -> Dropout(0.3) -> Dense(1, Sigmoid)
```

### Why These Choices?

| Component | Choice | Reason |
|-----------|--------|--------|
| Hidden activation | ReLU | No vanishing gradient, computationally cheap |
| Output activation | Sigmoid | Outputs probability 0-1 for binary classification |
| Loss function | Binary Crossentropy | Standard for binary classification |
| Optimizer | Adam | Adaptive learning rates, fast convergence |
| Regularization | Dropout + L2 | Prevents overfitting, improves generalization |

### Hyperparameter Tuning
Tested 4 configurations varying layer sizes, dropout rates, L2 strength, learning rate, and batch size.

## Evaluation

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve with AUC
- Precision-Recall Curve

### Business Interpretation

| Error Type | Meaning | Cost |
|------------|---------|------|
| False Positive | Predicted churn, customer stayed | Moderate - wasted retention budget |
| False Negative | Predicted stay, customer churned | **High** - lost revenue and lifetime value |

In telecom, False Negatives are more costly. Consider lowering the threshold to improve recall.

## Model Deployment

```python
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('churn_prediction_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

scaled_data = scaler.transform(new_customer_data)
churn_probability = model.predict(scaled_data)
```

## License

MIT License
