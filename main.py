import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
import utils

# Creating a DataFrame from the dataset already prepared
data = pd.read_csv('datasets/data.csv')

# Preprocess data
X_train_std, X_test_std, y_train, y_test = utils.preprocess_data(data)

## RANDOM FOREST
# Create a Gaussian Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators=100)
# Train the model using the training sets
model_rf.fit(X_train_std,y_train)
# and get predictions for the test
y_pred = model_rf.predict(X_test_std)
# Model Accuracy, how often is the classifier correct?
#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print('RANDOM FOREST')
print(metrics.classification_report(y_test, y_pred))

# Feature importance
feature_imp = pd.Series(model_rf.feature_importances_).sort_values(ascending=False)
utils.visualize_feature_importance(feature_imp, 'Random Forest')

## Gradient Boosting could be a step forward Random Forest for more complex problems
# sklearn.ensemble.GradientBoostingClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

## LOGISTIC REGRESSION
model_lr = LogisticRegression()
# Train the model using the training sets
model_lr.fit(X_train_std, y_train)
# and get predictions for the test
y_pred = model_lr.predict(X_test_std)

# Model Accuracy, how often is the classifier correct?
#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))# logistic regression is really affected by the normalization

# Get feature importance from the trained model
coefficients = np.abs(model_lr.coef_[0])

feature_imp = pd.Series(coefficients).sort_values(ascending=False)
utils.visualize_feature_importance(feature_imp, 'Logistic Regression')


## SUPPORT VECTOR MACHINE
model_svm = SVC(kernel='linear')  # feature coefficients are only available when using a linear kernel
model_svm.fit(X_train_std, y_train)

y_pred = model_svm.predict(X_test_std)

# Model Accuracy, how often is the classifier correct?
#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))# logistic regression is really affected by the normalization


# Get feature importance from the trained model
coefficients = np.abs(model_svm.coef_[0])

# Get feature importance from the trained model
feature_imp = pd.Series(coefficients).sort_values(ascending=False)
utils.visualize_feature_importance(feature_imp, 'Support Vector Machine')