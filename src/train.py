import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("dataset/main.csv")

# Define feature columns and the target column
X = data[['domain']]
data['target'] = data['target'].map({True: 1, False: 0})
y = data['target']

# Feature extraction: Convert domain names into numerical features
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))  # Using character n-grams
X_transformed = vectorizer.fit_transform(X['domain'])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100}%")

# Save the model
joblib.dump(model, "model/model.pkl")

# Save the vectorizer
joblib.dump(vectorizer, "model/vectorizer.pkl")
