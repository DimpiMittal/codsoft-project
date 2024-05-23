import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load and preprocess the data
credited_card_df = pd.read_csv('creditcard.csv')

# Separate the legitimate and fraudulent transactions
legit = credited_card_df[credited_card_df.Class == 0]
fraud = credited_card_df[credited_card_df.Class == 1]

# Sample the legitimate transactions to match the number of fraudulent transactions
legit_sample = legit.sample(n=fraud.shape[0])
credited_card_df = pd.concat([legit_sample, fraud], axis=0)

# Split the data into features and target
x = credited_card_df.drop('Class', axis=1)
y = credited_card_df['Class']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=10000, solver='sag')
model.fit(x_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(x_train), y_train)
test_acc = accuracy_score(model.predict(x_test), y_test)

# Display model performance
print(f'Training Accuracy: {train_acc}')
print(f'Test Accuracy: {test_acc}')

# Web app
st.title("Credit Card Fraud Detection Model")

st.write("Enter the 30 feature values separated by commas:")

# Define a text area for input without default values
input_text = st.text_area("Feature values", value="")

submit = st.button("Submit")

if submit:
    try:
        # Split the input text by commas and convert to a numpy array
        input_values = np.array([float(x) for x in input_text.split(',')])
        
        # Check if the correct number of features are provided
        if len(input_values) != x.shape[1]:
            st.write(f"Please enter exactly {x.shape[1]} values.")
        else:
            # Reshape and make prediction
            features = input_values.reshape(1, -1)
            prediction = model.predict(features)
            if prediction[0] == 0:
                st.write("Legitimate Transaction")
            else:
                st.write("Fraudulent Transaction")
    except ValueError:
        st.write("Please enter valid numerical values separated by commas.")
