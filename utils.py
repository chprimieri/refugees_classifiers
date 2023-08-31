import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    # Transform string labels in numerical
    le = LabelEncoder()
    le.fit(["Northern Africa", "Eastern Africa", "Middle Africa", "Southern Africa","Western Africa",
            "Caribbean", "Central America", "South America", "Northern America", "Central Asia",
            "Eastern Asia", "South-eastern Asia", "Southern Asia", "Western Asia", "Eastern Europe",
            "Northern Europe", "Southern Europe", "Western Europe", "Australia and New Zealand",
            "Melanesia", "Micronesia", "Polynesia"])
    data['Num Origin Region'] = le.transform(data[['Origin Region']].values.ravel())
    data['Num Asylum Region'] = le.transform(data[['Asylum Region']].values.ravel())

    # Split data into features and results (for the regression task)
    # Features
    X = data[['Num Origin Region','Num Asylum Region','Distance (Km)','HDI asylum','LE asylum',
              'EYS asylum','MYS asylum','GNIPC asylum','GDI asylum','GII asylum','PHDI asylum',
              'HDI diff','LE diff','EYS diff','MYS diff','GNIPC diff','GDI diff','GII diff',
              'PHDI diff']]  
    # Results
    y = data[['Refugees']].values.ravel()

    # Split dataset into training set and test set
    # 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  

    # Normalize data, assumes data is normally distributed
    scaler = StandardScaler()
    # Initialize the scaler with the training data
    scaler.fit(X_train)
    # Apply to train data
    X_train_std = scaler.transform(X_train) 
    # Apply to test data 
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test

def visualize_original_vs_predicted(y_test, y_pred, title):
    x_ax = range(len(y_test))

    # Plot chart
    plt.figure(figsize=(14, 6))
    plt.plot(x_ax, y_test, linewidth=1, label="True Values")
    plt.plot(x_ax, y_pred, linewidth=1.1, label="Predicted Values")

    # Add labels to chart
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Refugees')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.ticklabel_format(style='plain')
    plt.savefig('figures/' + title + '_original_predicted.png')

def visualize_feature_importance(feature_imp, title):
    # Plot chart
    plt.figure(figsize=(14, 6))
    sns.barplot(x=feature_imp, y=feature_imp.index)

    # Add labels to chart
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(title)
    plt.savefig('figures/' + title + '_feature_importance.png')
