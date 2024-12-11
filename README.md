# Loan Approval Prediction

## Abstract

Loan approval is a critical process in the banking and financial sector, impacting individuals’ abilities to achieve their dreams, secure homes, pursue education, or address emergencies. For international students and individuals stepping into a new phase of their lives, access to loans can be a stepping stone to personal and professional growth. However, the loan approval process is governed by stringent eligibility conditions, often creating challenges for applicants.

This project aims to develop a predictive model for loan approval, leveraging data-driven approaches to streamline the decision-making process for banks and licensed financial institutions. By analyzing a sample loan dataset, the project identifies key factors influencing loan eligibility, such as income, credit history, employment type, and debt-to-income ratio. Machine learning techniques are employed to create a robust, accurate, and interpretable model that predicts whether a loan application is likely to be approved.

This solution offers a dual benefit: it helps financial institutions optimize their loan approval processes while providing applicants with a clearer understanding of their eligibility. The project not only addresses the practical challenges of loan approvals but also contributes to the broader goal of making financial support more accessible to people across various economic and cultural backgrounds.

This problem is of great interest to me because I am currently pursuing a Master’s degree in Computer Science at the University of Texas at Dallas. This project aligns with my academic interests because I was challenged with loan approval back in my home country.

## Problem Statement

Through different factors like income, credit history, employment type, and debt-to-income ratio, the model predicts whether a loan application is likely to be approved.

## Classification Models Implemented

The following classification models were used in the project:
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **XGBoost**
5. **SVM**
6. **KNN**
7. **Naive Bayes**
8. **AdaBoost**
9. **Gradient Boosting**
10. **Neural Network**

## Project Type

This is a **classification** problem, where the goal is to predict a binary outcome (loan approval/rejection) based on various input features.

## Project Implementation Characteristics

- **Dataset Source**: The dataset is sourced from Kaggle: [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/subhamjain/loan-approval-prediction).
- **Data Analysis**: Comprehensive study of dataset characteristics, including correlations, label and feature distributions, missing values, imbalance, and mean/variance of features.
- **Machine Learning Algorithms**: A thorough evaluation of different machine learning algorithms (linear models, decision trees, boosted trees, random forests, neural networks, etc.).
- **Evaluation Metrics**: Use of meaningful evaluation metrics such as accuracy, precision, recall, F1-score, etc.
- **Model Analysis**: Analysis of what works and what does not work, identifying best-performing models.
- **Lessons Learned**: Insights gained from the project and how it contributed to a better understanding of machine learning techniques for financial applications.

## Dataset Overview

The dataset contains 45,000 rows and 14 columns. It includes 13 features and 1 target variable (`loan_status`), where the target is binary:
- `1`: Approved
- `0`: Rejected

### Dataset Columns:

| Column                         | Description                                      | Type        |
|--------------------------------|--------------------------------------------------|-------------|
| `person_age`                   | Age of the person                                | Float       |
| `person_gender`                | Gender of the person                             | Categorical |
| `person_education`             | Highest education level                          | Categorical |
| `person_income`                | Annual income                                    | Float       |
| `person_emp_exp`               | Years of employment experience                   | Integer     |
| `person_home_ownership`        | Home ownership status (e.g., rent, own, mortgage)| Categorical |
| `loan_amnt`                    | Loan amount requested                            | Float       |
| `loan_intent`                  | Purpose of the loan                              | Categorical |
| `loan_int_rate`                | Loan interest rate                               | Float       |
| `loan_percent_income`          | Loan amount as a percentage of annual income     | Float       |
| `cb_person_cred_hist_length`   | Length of credit history in years                | Float       |
| `credit_score`                 | Credit score of the person                       | Integer     |
| `previous_loan_defaults_on_file` | Indicator of previous loan defaults              | Categorical |
| `loan_status` (target)         | Loan approval status (1 = approved; 0 = rejected)| Integer     |

## Implemented Machine Learning Models

### 1. **Decision Tree**
The decision tree is a supervised learning algorithm that makes predictions by recursively partitioning the input space into smaller regions. Stopping criteria such as maximum depth, minimum samples per leaf, and impurity threshold help define the tree's growth.

### 2. **Logistic Regression**
Logistic regression is a statistical method used to model the relationship between a binary dependent variable and one or more independent variables. The logistic regression model predicts the probability of loan approval based on the predictor variables.

### 3. **Random Forest**
Random Forest is an ensemble method that combines multiple decision trees to make more accurate predictions. It is known for its ability to handle large datasets and provide better generalization than a single decision tree.

### 4. **Support Vector Machine (SVM)**
SVM is a supervised learning algorithm that finds the optimal hyperplane to separate different classes in the data. It works well with high-dimensional data and non-linear separability.

### 5. **K-means Clustering**
K-means is a clustering algorithm that groups similar data points into clusters. While it is primarily used for unsupervised learning, it was applied here to analyze patterns and group loan applicants based on features.

### 6. **Bagging and Boosting (Ensemble Methods)**
Both bagging and boosting are ensemble techniques. Bagging (Bootstrap Aggregating) improves the performance by aggregating the results of multiple models trained on different subsets of data. Boosting, on the other hand, sequentially improves model predictions by focusing on the errors made by previous models.

## Conclusion

This project aimed to develop an efficient model for predicting loan approvals using machine learning techniques. By exploring various models such as Decision Trees, Logistic Regression, Random Forest, and more, the project contributes towards making the loan approval process more transparent and accessible for individuals seeking financial support.

The outcome of this project can benefit both financial institutions and applicants by streamlining loan approval processes and improving decision-making accuracy.

## Lessons Learned

- **Feature Engineering**: Proper feature selection and preprocessing (e.g., handling categorical variables) is critical for model performance.
- **Model Selection**: Ensemble models such as Random Forest and Gradient Boosting provide better results than individual models like Decision Trees.
- **Evaluation Metrics**: Accuracy alone is not sufficient for evaluating classification models. Precision, recall, and F1-score provide a more complete picture, especially when dealing with imbalanced data.

## Future Work

In the future, we plan to:
- Explore additional features such as applicant's previous loan history or employment stability.
- Fine-tune the hyperparameters of the models to improve performance.
- Implement additional models like XGBoost and Neural Networks for enhanced prediction accuracy.

## References
- Kaggle Loan Approval Prediction Dataset: [Loan Approval Prediction](https://www.kaggle.com/datasets/subhamjain/loan-approval-prediction)

# Loan-Approval-Prediction
