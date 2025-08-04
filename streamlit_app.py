import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time

# Load and prepare data
df = pd.read_csv("Hours and Scores.csv")
df = df.drop('Unnamed: 0', axis=1)

X = df[['Hours']]
y = df['Scores']

# Train models
model = LinearRegression()
model.fit(X, y)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Streamlit App
st.set_page_config(page_title="📚 Score Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎓 Student Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("#### 📘 Enter the number of hours you studied to predict your exam score:")

hours = st.slider("📊 Hours Studied", 0.0, 12.0, 1.0, 0.25)

col1, col2 = st.columns(2)

with col1:
    if st.button("🧮 Predict (Simple)"):
        with st.spinner("Calculating..."):
            time.sleep(1)
            pred = model.predict([[hours]])
        st.success(f"📈 Predicted Score: `{pred[0]:.2f}`")

with col2:
    if st.button("📐 Predict (Polynomial)"):
        with st.spinner("Calculating..."):
            time.sleep(1)
            poly_pred = poly_model.predict(poly.transform([[hours]]))
        st.success(f"🧾 Polynomial Score: `{poly_pred[0]:.2f}`")

st.markdown("---")
st.caption("🛠️ Powered by Linear & Polynomial Regression Models")

