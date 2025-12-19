import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classification System")
st.write("Internship Machine Learning Project using Scikit-learn")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Iris.csv")
    return data

df = load_data()

# -------------------------------
# Dataset Overview
# -------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.write("**Dataset Shape:**", df.shape)

# -------------------------------
# Preprocessing
# -------------------------------
X = df.drop(columns=["Id", "Species"])
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"✅ **Accuracy:** {accuracy * 100:.2f}%")

# Classification Report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.write("Train Accuracy:", model.score(X_train, y_train))
st.write("Test Accuracy:", model.score(X_test, y_test))

# Confusion Matrix
st.subheader("🔍 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("🌼 Predict Iris Species")

sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("🔮 Predict Species"):
    prediction = model.predict(input_data)[0]
    st.success(f"🌸 Predicted Iris Species: **{prediction}**")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("📌 **Internship Project | Machine Learning | Iris Dataset**")
