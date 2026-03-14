import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

def load_data(data_path='data/loan_data.csv'):
    """Load the raw loan data."""
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df, training=False, expected_columns=None):
    """Preprocess the data: handle outliers, encode categorical, etc."""
    # Remove age outliers (>70) only for training data
    if training:
        df = df[df['person_age'] <= 70].copy()

    # Encode categorical variables
    categorical_cols = ['person_gender', 'person_education', 'person_home_ownership',
                       'loan_intent', 'previous_loan_defaults_on_file']

    # Label encoding for binary, one-hot for multi-class
    df['person_gender'] = df['person_gender'].map({'male': 1, 'female': 0})
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})

    # One-hot encoding for multi-class
    df = pd.get_dummies(df, columns=['person_education', 'person_home_ownership', 'loan_intent'],
                       drop_first=True)

    # Ensure all expected columns are present (for prediction)
    if expected_columns is not None:
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        # Reorder columns to match training
        df = df[expected_columns]

    return df

def split_data(df, target='loan_status', test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, numerical_cols):
    """Scale numerical features."""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    # Example usage
    df = load_data()
    df_processed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_processed)

    # Save processed data
    os.makedirs('data', exist_ok=True)
    df_processed.to_csv('data/df_processed.csv', index=False)

    print("Data preprocessing completed.")
    print(f"Processed data shape: {df_processed.shape}")