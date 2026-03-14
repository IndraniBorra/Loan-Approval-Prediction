import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib
import os
from preprocess import load_data, preprocess_data, split_data, scale_features

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        }

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()

        results[name] = {
            'model': model,
            'metrics': metrics
        }

        print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    return results

def save_models(results, models_dir='models'):
    """Save trained models."""
    os.makedirs(models_dir, exist_ok=True)
    for name, result in results.items():
        filename = f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(result['model'], os.path.join(models_dir, filename))

def main():
    # Load and preprocess data
    df = load_data()
    df_processed = preprocess_data(df)

    # Define numerical columns for scaling
    numerical_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                     'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']

    # Split data
    X_train, X_test, y_train, y_test = split_data(df_processed)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, numerical_cols)

    # Train models
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Save models
    save_models(results)

    # Print summary
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    for name, result in results.items():
        metrics = result['metrics']
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  CV Mean Accuracy: {metrics['cv_mean']:.4f}")
        print(f"  CV Std Dev: {metrics['cv_std']:.4f}")

    # Save results to CSV
    results_df = pd.DataFrame({
        name: {
            'Accuracy': result['metrics']['accuracy'],
            'Precision': result['metrics']['precision'],
            'Recall': result['metrics']['recall'],
            'F1 Score': result['metrics']['f1'],
            'ROC AUC': result['metrics']['roc_auc'] or 0,
            'CV Mean': result['metrics']['cv_mean'],
            'CV Std': result['metrics']['cv_std']
        }
        for name, result in results.items()
    }).T

    results_df.to_csv('reports/model_comparison.csv')
    print("\nResults saved to reports/model_comparison.csv")

if __name__ == "__main__":
    main()