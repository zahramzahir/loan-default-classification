# Model Evaluation Report

## 1. Introduction

This project uses machine learning techniques to predict whether or not a loan will default. I used two classification models: Logistic Regression and a Decision Tree. I also performed Exploratory Data Analysis to explore the data before building the models.

## 2. Data Cleaning and Preparation

The raw dataset contained approximately 30,000 rows, so I used the first 500 rows to keep the project manageable.

I selected relevant columns, removed missing values and duplicates, and used one-hot encoding to convert categorical variables into numerical values.

The data was then split into 80% training data and 20% testing data.

## 3. Exploratory Data Analysis

Before building the models, I looked at the proportion of loans based on their default status. I then compared the average income and loan amount between the groups.

### Findings

* 43.84% of the loans did not default.
* 56.16% of the loans defaulted.
* The average income was much higher for borrowers who did not default.
* The average loan amount was similar between the two groups.

## 4. Models

I used two machine learning models:

* **Logistic Regression**, commonly used for binary classification.
* **Decision Tree**, which splits the data to make predictions.

## 5. Evaluation

I used the following metrics to evaluate the models:

* **Precision:** the proportion of predicted defaults which were correctly identified.
* **Recall:** the proportion of actual defaults which were correctly identified.
* **F1 Score:** a balanced score of precision and recall.
* **ROC-AUC:** a measure of how well the model distinguishes between the two classes (default and no default).

I also used 5-fold cross-validation to observe how the models performed across different subsets of the data.

## 6. Results

| Metric    | Logistic Regression | Decision Tree |
| --------- | ------------------: | ------------: |
| Precision |               0.815 |         0.820 |
| Recall    |               0.914 |         0.862 |
| F1 Score  |       0.866 ± 0.017 | 0.885 ± 0.033 |
| ROC-AUC   |       0.931 ± 0.025 | 0.937 ± 0.018 |

## 7. Interpretation

Logistic Regression had a higher recall, whilst the Decision Tree had a higher precision, F1 Score and ROC-AUC.

The higher recall suggests that the Logistic Regression model was able to identify more of the loans which defaulted than the Decision Tree. The Decision Tree had higher F1 Score and ROC-AUC values, showing stronger performance according to those measures.

## 8. Limitations

Since only 500 rows were used alongside a limited number of columns, the results may differ when using the full dataset.

This is a beginner project, and there is room for improved testing, additional features and further evaluation.

## 9. Conclusion

Both models produced fairly similar results and performed differently across the metrics. This shows the trade-off between the models depending on which metric is considered most important.
