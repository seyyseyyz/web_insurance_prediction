import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sklearn

print(sklearn.__version__)

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="wide")

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("insurance.csv")

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("🧾 User Information")

age = st.sidebar.slider("Age", 18, 64, 25)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox(
    "Region", ["southwest", "southeast", "northwest", "northeast"]
)

predict_button = st.sidebar.button(
    "💊 Predict Insurance Cost", use_container_width=True
)

# -------------------------
# Convert input
# -------------------------
sex_num = 1 if sex == "male" else 0
smoker_num = 1 if smoker == "yes" else 0
region_map = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
region_num = region_map[region]

input_data = [[age, sex_num, bmi, children, smoker_num, region_num]]

# -------------------------
# TOP SECTION (UPDATED)
# -------------------------
st.title("💊 Medical Insurance Cost Prediction")
st.subheader("Estimate your medical insurance charges")

st.markdown(
    """
This web application uses **Machine Learning** to estimate your medical insurance cost.

👉 Please enter your information in the sidebar.
"""
)

# 🔥 FULL WIDTH IMAGE
st.image("insurance.jpeg", use_container_width=True)

# -------------------------
# Prediction section
# -------------------------
st.markdown("---")
st.header("🔮 Prediction Result")

if predict_button:
    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div style="
            background-color:#eef8f0;
            padding:20px;
            border-radius:10px;
            border:1px solid #cde8d1;
        ">
            <h3 style="color:#1f2937;">Estimated Insurance Cost</h3>
            <h1 style="color:#1e8e3e;">$ {prediction:,.2f}</h1>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.success("Prediction generated successfully.")

else:
    st.info("Fill in the information and click 'Predict Insurance Cost'.")

# -------------------------
# EDA section
# -------------------------
st.markdown("---")
st.header("📊 Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Charges")
    fig, ax = plt.subplots()
    ax.hist(df["charges"], bins=30)
    ax.set_xlabel("Charges")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.subheader("Age vs Charges")
    fig, ax = plt.subplots()
    ax.scatter(df["age"], df["charges"])
    ax.set_xlabel("Age")
    ax.set_ylabel("Charges")
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.subheader("BMI vs Charges")
    fig, ax = plt.subplots()
    ax.scatter(df["bmi"], df["charges"])
    ax.set_xlabel("BMI")
    ax.set_ylabel("Charges")
    st.pyplot(fig)

with col4:
    st.subheader("Smoker vs Charges")
    smoker_yes = df[df["smoker"] == "yes"]["charges"]
    smoker_no = df[df["smoker"] == "no"]["charges"]
    fig, ax = plt.subplots()
    ax.boxplot([smoker_yes, smoker_no], labels=["Yes", "No"])
    ax.set_xlabel("Smoker")
    ax.set_ylabel("Charges")
    st.pyplot(fig)

# -------------------------
# Feedback section
# -------------------------
st.markdown("---")
st.header("💬 User Feedback")

rating = st.slider("Rate this app", 1, 5, 3)
feedback = st.text_area("Write your feedback here")

if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")
