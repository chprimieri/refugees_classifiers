import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn import metrics
import utils

# Creating a DataFrame from the dataset already prepared
data = pd.read_csv('datasets/data.csv')

# Preprocess data
X_train_std, X_test_std, y_train, y_test = utils.preprocess_data(data)


## RANDOM FOREST ##
# Create a Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=100)

# Train the model using the training sets
model_rf.fit(X_train_std, y_train)

print("\nRANDOM FOREST")

# Get predictions for the test
y_pred = model_rf.predict(X_test_std)

# Model Accuracy, how often is the regression correct?
print("Accuracy R2 Score: ", metrics.r2_score(y_test, y_pred))

# Median Absolute Error of the model
print('Median Absolute Error:', round(metrics.median_absolute_error(y_test, y_pred), 0))

# Visualize the Original Vs Predicted Data
# utils.visualize_original_vs_predicted(y_test, y_pred)

# Feature importance
feature_imp = pd.Series(model_rf.feature_importances_, index=data[['Origin Region','Asylum Region','Distance (Km)','HDI asylum','LE asylum','EYS asylum',
              'MYS asylum','GNIPC asylum','GDI asylum','GII asylum','PHDI asylum','HDI diff',
              'LE diff','EYS diff','MYS diff','GNIPC diff','GDI diff','GII diff','PHDI diff']].columns)
# print(feature_imp.head(50))

# utils.visualize_feature_importance(feature_imp, 'Random Forest')


## ADA BOOST ##
# Create an Ada Boost Regressor 
model_ab = AdaBoostRegressor(random_state=0, n_estimators=100)

# Train the model using the training sets
model_ab.fit(X_train_std, y_train)

# Get predictions for the test
y_pred = model_ab.predict(X_test_std)

print("\nADA BOOST")

# Model Accuracy, how often is the regression correct?
print("Accuracy R2 Score: ", metrics.r2_score(y_test, y_pred))

# Median Absolute Error of the model
print('Median Absolute Error:', round(metrics.median_absolute_error(y_test, y_pred), 0))

# Visualize the Original Vs Predicted Data
# utils.visualize_original_vs_predicted(y_test, y_pred)

# Feature importance
feature_imp = pd.Series(model_ab.feature_importances_, index=data[['Origin Region','Asylum Region','Distance (Km)','HDI asylum','LE asylum','EYS asylum',
              'MYS asylum','GNIPC asylum','GDI asylum','GII asylum','PHDI asylum','HDI diff',
              'LE diff','EYS diff','MYS diff','GNIPC diff','GDI diff','GII diff','PHDI diff']].columns)
# print(feature_imp.head(50))
# utils.visualize_feature_importance(feature_imp, 'Ada Boost')


## GRADIENT BOOSTING ##
# Create an Histogram-based Gradient Boosting Regressor 
model_gb = GradientBoostingRegressor(random_state=0)

# Train the model using the training sets
model_gb.fit(X_train_std, y_train)

print("\nGRADIENT BOOSTING")

# Get predictions for the test
y_pred = model_gb.predict(X_test_std)

# Model Accuracy, how often is the regression correct?
print("Accuracy R2 Score: ", metrics.r2_score(y_test, y_pred))

# Median Absolute Error of the model
print('Median Absolute Error:', round(metrics.median_absolute_error(y_test, y_pred), 0))

# Visualize the Original Vs Predicted Data
# utils.visualize_original_vs_predicted(y_test, y_pred)

# Feature importance
feature_imp = pd.Series(model_gb.feature_importances_, index=data[['Origin Region','Asylum Region','Distance (Km)','HDI asylum','LE asylum','EYS asylum',
              'MYS asylum','GNIPC asylum','GDI asylum','GII asylum','PHDI asylum','HDI diff',
              'LE diff','EYS diff','MYS diff','GNIPC diff','GDI diff','GII diff','PHDI diff']].columns)
# print(feature_imp.head(50))
# utils.visualize_feature_importance(feature_imp, 'Histogram-based Gradient Boosting')