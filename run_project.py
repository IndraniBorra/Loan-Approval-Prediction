#!/usr/bin/env python3
"""
Loan Approval Prediction Project Runner

This script runs the prediction workflow:
- Installs dependencies once (if needed)
- Preprocesses data once (if needed)
- Trains models once (if needed)
- Prompts for user input to predict loan approval

Usage:
    python run_project.py [--force-train] [--skip-install]

Options:
    --force-train: Force retraining of models
    --skip-install: Skip dependency check/installation
"""

import argparse
import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import pandas
        import sklearn
        import joblib
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install dependencies if not already installed."""
    if check_dependencies():
        print("✓ Dependencies already installed, skipping...")
        return
    run_command("pip install -r requirements.txt", "Installing dependencies")

def run_command(command, description):
    """Run a shell command and print status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}: {e}")
        print(f"Output: {e.output}")
        sys.exit(1)

def check_data_processed():
    """Check if processed data exists."""
    return os.path.exists('data/df_processed.csv')

def check_models_trained():
    """Check if models are trained."""
    model_files = ['models/logistic_regression.pkl', 'models/random_forest.pkl', 'models/svm.pkl']
    return all(os.path.exists(f) for f in model_files)

def main():
    parser = argparse.ArgumentParser(description='Run the full Loan Approval Prediction workflow')
    parser.add_argument('--interactive', action='store_true', help='Run prediction in interactive mode')
    parser.add_argument('--skip-install', action='store_true', help='Skip dependency installation')
    parser.add_argument('--force-train', action='store_true', help='Force retraining of models')
    args = parser.parse_args()

    # Change to project root if not already
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Step 1: Install dependencies (unless skipped)
    if not args.skip_install:
        install_dependencies()

    # Step 2: Preprocess data (if not already done)
    if not check_data_processed():
        run_command("python3 src/preprocess.py", "Preprocessing data")
    else:
        print("✓ Data already processed, skipping...")

    # Step 3: Train models (if not already done or forced)
    if not check_models_trained() or args.force_train:
        run_command("python3 src/train_models.py", "Training models")
    else:
        print("✓ Models already trained, skipping...")

    # Step 4: Make predictions (always interactive for user input)
    predict_cmd = "python3 src/predict.py --interactive"
    print(f"Running: Making predictions")
    try:
        subprocess.run(predict_cmd, shell=True, check=True)
        print("✓ Making predictions completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in Making predictions: {e}")
        sys.exit(1)

    print("\n🎉 Prediction completed! Check above for results.")

if __name__ == "__main__":
    main()