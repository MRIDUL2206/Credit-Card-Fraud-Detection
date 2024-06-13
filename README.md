# Credit-Card-Fraud-Detection

Welcome to the Credit Card Fraud Detection project! This repository contains a comprehensive implementation of a machine learning model to detect fraudulent credit card transactions using the Decision Tree algorithm. The dataset for this project has been sourced from Kaggle and includes a variety of transaction data that will help in identifying fraudulent activities.

## Project Overview

Credit card fraud is a significant issue that affects both consumers and financial institutions. By leveraging machine learning techniques, we can develop models that accurately detect and prevent fraudulent transactions. This project utilizes the Decision Tree algorithm to classify transactions as either fraudulent or legitimate based on patterns and features present in the dataset.

## Dataset

The dataset used in this project is a CSV file obtained from Kaggle. It contains transaction details such as the amount, transaction date, and various anonymized features. The dataset is split into training and testing sets to evaluate the model's performance.

## Libraries and Frameworks

The following libraries and frameworks have been used in this project:

- `pandas` for data manipulation and analysis
- `scikit-learn` for model building and evaluation
  - `train_test_split` for splitting the dataset
  - `DecisionTreeClassifier` for building the Decision Tree model
  - `accuracy_score`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score` for evaluating model performance

## Model Performance

The Decision Tree model achieved an impressive accuracy of **99.92%**, demonstrating its effectiveness in detecting fraudulent transactions. Here are some key metrics:

- **Accuracy**: 99.92%
- **Precision**: High precision indicates a low false positive rate.
- **Recall**: High recall indicates a low false negative rate.
- **F1 Score**: Balance between precision and recall.

## Usage

To run this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/MRIDUL2206/Credit-Card-Fraud-Detection.git
    ```

2. Open the jupyter file.

3. Download any dataset for model training. The link for the dataset i used is attached below:
    ```sh
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    ```

4. Install the required libraries:
    ```sh
    pip install pandas scikit-learn
    ```

5. Run the model by running all cells present in the ipynb file.

## Conclusion

This project demonstrates the application of Decision Trees in detecting credit card fraud with high accuracy. Feel free to explore the code and experiment with different parameters and models to further improve the results.Finally I have been able to achieve over 99% accuracy on a test dataset. Contributions and feedback are welcome!
