import joblib

# Load the model
model = joblib.load("model/model.pkl")

# Load the vectorizer
vectorizer = joblib.load("model/vectorizer.pkl")

# Predict for a new domain
input_domain = input("Input Domain: ")
input_transformed = vectorizer.transform([input_domain])
prediction = model.predict(input_transformed)

if prediction[0] == 1:
    print(f"The domain '{input_domain}' is likely a scam.")
else:
    print(f"The domain '{input_domain}' is likely safe.")
