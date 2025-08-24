# 🏥 Medical Insurance Cost Prediction – Linear Regression

A predictive model built using Linear Regression to estimate medical insurance costs based on demographic and health-related features. This project includes the full ML pipeline—data exploration, preprocessing, modeling, evaluation—with a focus on interpretability and simplicity.

👉 Live Demo: [Medical Insurance Cost Prediction App](https://medicalinsurancecostpredictionlinearregression-ibya7mhtue5zb27.streamlit.app/)

## 📌 Project Overview

Objective: Predict individual insurance charges using attributes like age, BMI, smoking status, region, etc.

Model: Linear Regression—chosen for its simplicity and interpretability in regression tasks.

Includes:

Exploratory Data Analysis (EDA) & visualizations

Feature engineering and preprocessing

Model training and validation

Evaluation using RMSE, R²

Deployed with Streamlit

## 📂 Repository Structure

/Medical_Insurance_Cost_Prediction_Linear_Regression

│── insurance.csv                        # Dataset

│── Medical_Insurance_Cost_Prediction.ipynb  # Jupyter notebook

│── app.py                               # Streamlit app

│── requirements.txt                     # Dependencies

│── README.md                            # Documentation

## 🚀 Getting Started

### 1. Clone the Repository

git clone https://github.com/abhinav744/Medical_Insurance_Cost_Prediction_Linear_Regression.git

cd Medical_Insurance_Cost_Prediction_Linear_Regression

### 2. (Optional) Create & Activate a Virtual Environment

python -m venv venv

source venv/bin/activate       # macOS/Linux

venv\Scripts\activate          # Windows

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run the Streamlit App

streamlit run app.py

## 📊 Insights & Performance

Key predictors: Smoking status, BMI, and age

Evaluation metrics:

RMSE ≈ $6,000–$6,500

R² ≈ 0.75–0.80

## 🔮 Future Enhancements

Add regularization (Ridge, Lasso)

Compare with tree-based models (Random Forest, XGBoost)

More visual regression diagnostics

Enhance Streamlit app with better UI & insights
