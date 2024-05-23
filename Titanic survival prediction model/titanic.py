import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Read the dataset
df = pd.read_csv('titanic.csv')

# Data Exploration and Visualization
print(df.head(10))
print(df.shape)
print(df.describe())
print(df['Survived'].value_counts())

# Visualize survival rate by Pclass
sns.countplot(x='Survived', hue='Pclass', data=df)

# Visualize survival rate by Sex
sns.countplot(x='Survived', hue='Sex', data=df)

# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch']

# Handling Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)

# Encode categorical variables
labelencoder = LabelEncoder()
df['Sex'] = labelencoder.fit_transform(df['Sex'])

# Drop non-required columns
df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# Define features and target
X = df[['Pclass', 'Sex', 'Age', 'FamilySize']]
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model training
logistic_model = LogisticRegression(random_state=0)
logistic_model.fit(X_train, y_train)

# Model evaluation
y_pred = logistic_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Make predictions
def predict_survival(model, features):
    prediction = model.predict(features)[0]
    if prediction == 0:
        return "Not survived"
    else:
        return "Survived"

# Example prediction
test_features = [[2, 1, 30, 1]]  # Example features: Pclass=2, Sex=male, Age=30, FamilySize=1
print("Prediction:", predict_survival(logistic_model, test_features))
