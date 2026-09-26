# Binary Classification using logistic regression and decision trees

import pandas as pd 
# load the dataset
df = pd.read_csv("data/loan_dataset.csv")

# select a subset of rows and relevant columns
clean_df = df.iloc[:500, [0, 1, 2, 4, 6, 8]]

# remove missing values and duplicates
clean_df.dropna(inplace = True)
clean_df.drop_duplicates(inplace = True)

# one hot encoding to transform categorical data into binary features
new_df = pd.get_dummies(clean_df, columns = ['person_home_ownership', 'loan_intent'], dtype = int)

# EDA: target analysis
print(f"No default: {new_df['loan_status'].value_counts(normalize = True)[0]:.2%}")
print(f"Default: {new_df['loan_status'].value_counts(normalize=True)[1]:.2%}\n")

# EDA: bivariate analysis - features vs target
print(f"Average income — no default: {new_df.groupby('loan_status')['person_income'].mean()[0]:,.2f}")
print(f"Average income — default: {new_df.groupby('loan_status')['person_income'].mean()[1]:,.2f}\n")

print(f"Average loan amount — no default: {new_df.groupby('loan_status')['loan_amnt'].mean()[0]:,.2f}")
print(f"Average loan amount — default: {new_df.groupby('loan_status')['loan_amnt'].mean()[1]:,.2f}\n")

simple_df = new_df.iloc[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 12]]

# model preparation
y = simple_df["loan_status"]
X = simple_df.drop(columns = ["loan_status"])

# imports for logistic regression model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# create logistic regression pipeline
lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter = 1000)
)

lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

from sklearn.metrics import precision_score, recall_score

# evaluate test split
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)

# cross validation for logistic regression 
from sklearn.model_selection import StratifiedKFold, cross_validate

# 5 fold stratified cross validation
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
lr_scores = cross_validate(lr, X, y, cv = skf, scoring = ["f1", "roc_auc"])

# precision/ recall from test split, F1/ ROC-AUC from CV (mean ± std)
print(f"Logistic Regression Scores:\n"
      f"Precision: {lr_precision:.3f}\n"
      f"Recall: {lr_recall:.3f}\n"
      f"F1:      {lr_scores['test_f1'].mean():.3f} ± {lr_scores['test_f1'].std():.3f}\n"
      f"ROC-AUC: {lr_scores['test_roc_auc'].mean():.3f} ± {lr_scores['test_roc_auc'].std():.3f}\n"
)

from sklearn.tree import DecisionTreeClassifier

# define decision tree model
dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)

dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# evaluate test split
dt_precision = precision_score(y_test, y_pred_dt)
dt_recall = recall_score(y_test, y_pred_dt)

# 5 fold stratified cross validation
dt_scores = cross_validate(dt, X, y, cv = skf, scoring = ["f1", "roc_auc"])

# precision/ recall from test split, F1/ ROC-AUC from CV (mean ± std)
print(f"Decision Tree Scores:\n"
      f"Precision: {dt_precision:.3f}\n"
      f"Recall: {dt_recall:.3f}\n"
      f"F1:      {dt_scores['test_f1'].mean():.3f} ± {dt_scores['test_f1'].std():.3f}\n"
      f"ROC-AUC: {dt_scores['test_roc_auc'].mean():.3f} ± {dt_scores['test_roc_auc'].std():.3f}"
)
