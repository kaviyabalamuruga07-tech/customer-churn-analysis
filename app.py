import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data preprocessing
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.title("📊 Customer Churn Prediction App")

st.write("Enter Customer Details Below:")

monthly_charges = st.slider("Monthly Charges", 0, 150, 50)
tenure = st.slider("Tenure (Months)", 0, 72, 12)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "MonthlyCharges": [monthly_charges],
        "tenure": [tenure]
    })

    # Add missing columns with 0
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X.columns]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

