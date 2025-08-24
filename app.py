import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# -------------------------------
# Load and preprocess dataset
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("insurance.csv")
    data.replace({'sex':{'male':0, 'female':1}}, inplace=True)
    data.replace({'smoker':{'yes':0, 'no':1}}, inplace=True)
    data.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}}, inplace=True)
    return data

insurance_dataset = load_data()

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💰 Medical Insurance Cost Prediction")
st.markdown("This app predicts **insurance charges** based on health & lifestyle factors.")

# Sidebar for inputs
st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.radio("Sex", ("Male", "Female"))
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
children = st.sidebar.selectbox("Number of Children", [0,1,2,3,4,5])
smoker = st.sidebar.radio("Smoker", ("Yes", "No"))
region = st.sidebar.selectbox("Region", ("southeast", "southwest", "northeast", "northwest"))

# Encoding inputs
sex_val = 0 if sex == "Male" else 1
smoker_val = 0 if smoker == "Yes" else 1
region_dict = {"southeast":0, "southwest":1, "northeast":2, "northwest":3}
region_val = region_dict[region]

input_data = pd.DataFrame([[age, sex_val, bmi, children, smoker_val, region_val]], 
                          columns=["age", "sex", "bmi", "children", "smoker", "region"])

# Prediction
if st.sidebar.button("Predict Insurance Cost"):
    prediction = regressor.predict(input_data)
    st.success(f"💵 Estimated Insurance Cost: **USD {prediction[0]:.2f}**")

# -------------------------------
# Model performance metrics
# -------------------------------
st.subheader("📊 Model Performance")
train_pred = regressor.predict(X_train)
test_pred = regressor.predict(X_test)

r2_train = metrics.r2_score(Y_train, train_pred)
r2_test = metrics.r2_score(Y_test, test_pred)

st.write(f"✅ R² Score (Training): **{r2_train:.2f}**")
st.write(f"✅ R² Score (Testing): **{r2_test:.2f}**")

# Show dataset preview
with st.expander("🔎 View Dataset"):
    st.dataframe(insurance_dataset.head())
