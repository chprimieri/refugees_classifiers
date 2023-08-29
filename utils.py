import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(data):
    # Split data into features and labels (for the classification task)
    # Features
    X = data[['Refugees','Distance (Km)','HDI asylum','LE asylum','EYS asylum','MYS asylum',
              'GNIPC asylum','GDI asylum','GII asylum','PHDI asylum','HDI diff','LE diff',
              'EYS diff','MYS diff','GNIPC diff','GDI diff','GII diff','PHDI diff']]  
    # Labels
    y = data[['Year', 'Country of asylum', 'Asylum Region']]

    # Split dataset into training set and test set
    # 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  

    # Normalize data, assumes data is normally distributed (e.g. Gaussian with 0 mean and unit variance)
    scaler = StandardScaler()
    # Initialize the scaler with the training data
    scaler.fit(X_train)
    # Apply to train data
    X_train_std = scaler.transform(X_train) 
    # Apply to test data 
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test

def visualize_feature_importance(feature_imp, title):
    plt.figure(figsize=(14, 6))
    sns.barplot(x=feature_imp, y=feature_imp.index)

    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(title)
    plt.legend()
    plt.show()