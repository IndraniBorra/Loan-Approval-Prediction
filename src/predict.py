import argparse
import pandas as pd
import joblib
import numpy as np
from preprocess import preprocess_data

def load_model(model_name='gradient_boosting'):
    """Load a trained model."""
    model_path = f"models/{model_name}.pkl"
    model = joblib.load(model_path)
    return model

def predict_loan_approval(input_data, model_name='gradient_boosting'):
    """
    Predict loan approval for new data.

    input_data: dict with keys matching the original features
    """
    # Load training data to get expected columns
    df_train = pd.read_csv('data/df_processed.csv')
    expected_columns = [col for col in df_train.columns if col != 'loan_status']

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Preprocess
    df_processed = preprocess_data(df, training=False, expected_columns=expected_columns)

    # Load model
    model = load_model(model_name)

    # Predict
    prediction = model.predict(df_processed)
    probability = model.predict_proba(df_processed)[0][1] if hasattr(model, 'predict_proba') else None

    print(f"Debug: Raw prediction = {prediction[0]}, Probability = {probability}")  # Debug line

    result = {
        'prediction': int(prediction[0]),
        'approval_status': 'Approved' if prediction[0] == 1 else 'Rejected',
        'confidence': float(probability) if probability is not None else None
    }

    return result

def _prompt_float(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()
    return float(value) if value else default


def _prompt_int(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()
    return int(value) if value else default


def _prompt_str(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default


def _prompt_categorical(prompt, default, case='upper'):
    value = input(f"{prompt} [{default}]: ").strip()
    if not value:
        return default
    if case == 'upper':
        return value.upper()
    elif case == 'lower':
        return value.lower()
    elif case == 'title':
        return value.title()
    return value


def main():
    parser = argparse.ArgumentParser(description='Predict loan approval for a single applicant.')
    parser.add_argument('--model', default='random_forest', help='Model name to use (filename without .pkl)')
    parser.add_argument('--interactive', action='store_true', help='Prompt for input values interactively')
    args = parser.parse_args()

    if args.interactive:
        print("Enter applicant data (press Enter to use the default value):")
        sample_input = {
            'person_age': _prompt_float('Person age', 30.0),
            'person_gender': _prompt_categorical('Person gender (male/female)', 'female', 'lower'),
            'person_education': _prompt_categorical('Education (e.g. Bachelor)', 'Bachelor', 'title'),
            'person_income': _prompt_float('Annual income', 50000.0),
            'person_emp_exp': _prompt_int('Years of employment experience', 5),
            'person_home_ownership': _prompt_categorical('Home ownership (RENT/OWN/MORTGAGE)', 'RENT', 'upper'),
            'loan_amnt': _prompt_float('Loan amount requested', 10000.0),
            'loan_intent': _prompt_categorical('Loan intent (e.g. PERSONAL, EDUCATION)', 'PERSONAL', 'upper'),
            'loan_int_rate': _prompt_float('Loan interest rate', 12.5),
            'loan_percent_income': _prompt_float('Loan percent of income', 0.2),
            'cb_person_cred_hist_length': _prompt_float('Credit history length (years)', 5.0),
            'credit_score': _prompt_int('Credit score', 650),
            'previous_loan_defaults_on_file': _prompt_categorical('Previous loan defaults on file (Yes/No)', 'No', 'title')
        }
    else:
        sample_input = {
            'person_age': 30.0,
            'person_gender': 'female',
            'person_education': 'Bachelor',
            'person_income': 50000.0,
            'person_emp_exp': 5,
            'person_home_ownership': 'RENT',
            'loan_amnt': 10000.0,
            'loan_intent': 'PERSONAL',
            'loan_int_rate': 12.5,
            'loan_percent_income': 0.2,
            'cb_person_cred_hist_length': 5.0,
            'credit_score': 650,
            'previous_loan_defaults_on_file': 'No'
        }

    result = predict_loan_approval(sample_input, model_name=args.model)
    print("\nLoan Approval Prediction:")
    print(f"Status: {result['approval_status']}")
    if result['confidence'] is not None:
        print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    main()