import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(df):
    """Preprocesses the data by handling missing values and encoding categoricals."""

    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    numerical_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

# Load data and rename MachineID
train = pd.read_csv("data/train.csv").rename(columns={'MachineID': 'machineidentifier'})
test = pd.read_csv("data/test.csv").rename(columns={'MachineID': 'machineidentifier'})

# Preprocess
X = preprocess_data(train.drop('target', axis=1))
y = train['target']
test_processed = preprocess_data(test)  # Process test data separately

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Model Training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Model Evaluation
y_pred_val = rf_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_val)
print(f"Validation AUC: {auc}")

# Prediction on Test Set
y_pred_test = rf_model.predict_proba(test_processed)[:, 1]

# Create Submission File
submission = pd.DataFrame({'machineidentifier': test_processed['machineidentifier'], 'HasDetections': y_pred_test})
submission.to_csv('submission.csv', index=False)

print("Complete!  Submission file 'submission.csv' created.")