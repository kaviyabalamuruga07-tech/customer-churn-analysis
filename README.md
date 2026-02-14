# 📊 Customer Churn Analysis using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow?logo=scikitlearn)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 📌 Project Overview

Customer churn is a major problem in the telecom industry.  
This project builds a **Logistic Regression model** to predict whether a customer will leave the company.

The goal is to help businesses:
- Identify high-risk customers  
- Improve retention strategies  
- Reduce revenue loss  

---

## 🎯 Business Objective

Predict customer churn using historical telecom data to support data-driven decision making.

---

## 📂 Dataset

File Used: `telco_customer_churn_sample.csv`

### Key Features:
- Gender  
- SeniorCitizen  
- Tenure  
- MonthlyCharges  
- TotalCharges  
- Contract  
- InternetService  
- Churn (Target Variable)

---

## 📊 Exploratory Data Analysis (EDA)

- Checked missing values
- Analyzed churn distribution
- Calculated churn percentage
- Visualized:
  - Churn by Contract Type
  - Monthly Charges vs Churn

### 🔎 Key Insights
- Month-to-month contracts have higher churn rate
- Higher monthly charges slightly increase churn probability
- Long-term contracts reduce churn

---

## ⚙️ Machine Learning Workflow

1. Data Cleaning  
2. Dropped unnecessary column (`customerID`)  
3. Converted target variable (Yes → 1, No → 0)  
4. One-Hot Encoding using `pd.get_dummies()`  
5. Train-Test Split (80/20)  
6. Logistic Regression Model Training  
7. Model Evaluation  

---

## 📈 Model Performance

Example Output:

```
Model Accuracy: 0.68

Confusion Matrix:
[[92 14]
 [50 44]]
```

### Interpretation:
- 92 correct non-churn predictions
- 44 correct churn predictions
- Model can be improved with advanced algorithms

---

## 🚀 How to Run

### 1️⃣ Clone Repository
```
git clone https://github.com/kaviyabalamuruga07-tech/customer-churn-analysis.git
```

### 2️⃣ Install Dependencies
```
pip install pandas matplotlib seaborn scikit-learn
```

### 3️⃣ Run the Project
```
python main.py
```

---

## 📌 Future Improvements

- Apply Random Forest
- Use XGBoost
- Hyperparameter Tuning
- Handle class imbalance
- Deploy using Streamlit

---

## 👩‍💻 Author

**Kaviya Balamurugan**  
Aspiring Data Analyst | Machine Learning Enthusiast  

---

⭐ If you like this project, give it a star!

