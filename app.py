"""
Medical Insurance Cost Prediction
A Streamlit app for predicting medical insurance charges using a trained ML model.
"""
 
import logging
import pickle
from pathlib import Path
 
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PAGE_TITLE = "Medical Insurance Cost Prediction"
DATA_PATH = Path("insurance.csv")
MODEL_PATH = Path("model.pkl")
IMAGE_PATH = Path("insurance.jpeg")
 
REGION_MAP: dict[str, int] = {
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3,
}
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Data & model loading  (cached so they are loaded only once per session)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
 
 
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)
 
 
# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def encode_inputs(
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
) -> list[list]:
    """Encode raw user inputs into the numeric format the model expects."""
    return [[
        age,
        int(sex == "male"),
        bmi,
        children,
        int(smoker == "yes"),
        REGION_MAP[region],
    ]]
 
 
# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    """Render the sidebar inputs and return a dict of raw values."""
    st.sidebar.header("🧾 User Information")
    inputs = {
        "age":      st.sidebar.slider("Age", 18, 64, 25),
        "sex":      st.sidebar.selectbox("Sex", ["male", "female"]),
        "bmi":      st.sidebar.slider("BMI", 10.0, 50.0, 25.0, step=0.1),
        "children": st.sidebar.slider("Number of Children", 0, 5, 0),
        "smoker":   st.sidebar.selectbox("Smoker", ["yes", "no"]),
        "region":   st.sidebar.selectbox("Region", list(REGION_MAP)),
    }
    inputs["predict"] = st.sidebar.button(
        "💊 Predict Insurance Cost", use_container_width=True
    )
    return inputs
 
 
def render_prediction(model, input_data: list[list]) -> None:
    """Run inference and display the result."""
    st.header("🔮 Prediction Result")
    prediction: float = model.predict(input_data)[0]
    st.markdown(
        f"""
        <div style="
            background-color:#eef8f0;
            padding:20px;
            border-radius:10px;
            border:1px solid #cde8d1;
        ">
            <h3 style="color:#1f2937;">Estimated Insurance Cost</h3>
            <h1 style="color:#1e8e3e;">${prediction:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    logger.info("Prediction made: %.2f", prediction)
 
 
def _scatter(ax, x, y, xlabel: str) -> None:
    ax.scatter(x, y, alpha=0.5, s=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Charges ($)")
 
 
def render_eda(df: pd.DataFrame) -> None:
    """Render the Exploratory Data Analysis section."""
    st.header("📊 Exploratory Data Analysis")
 
    plots = [
        ("Distribution of Charges",  "hist"),
        ("Age vs Charges",            "age"),
        ("BMI vs Charges",            "bmi"),
        ("Smoker vs Charges",         "smoker"),
    ]
 
    cols = st.columns(2)
    for idx, (title, kind) in enumerate(plots):
        with cols[idx % 2]:
            st.subheader(title)
            fig, ax = plt.subplots()
 
            if kind == "hist":
                ax.hist(df["charges"], bins=30, edgecolor="white")
                ax.set_xlabel("Charges ($)")
                ax.set_ylabel("Count")
            elif kind in ("age", "bmi"):
                _scatter(ax, df[kind], df["charges"], kind.upper())
            elif kind == "smoker":
                groups = [
                    df.loc[df["smoker"] == "yes", "charges"],
                    df.loc[df["smoker"] == "no",  "charges"],
                ]
                ax.boxplot(groups, labels=["Smoker", "Non-smoker"])
                ax.set_ylabel("Charges ($)")
 
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)          # prevent memory leaks
 
 
def render_feedback() -> None:
    """Render the user feedback section."""
    st.header("💬 User Feedback")
    st.slider("Rate this app", 1, 5, 3, key="rating")
    st.text_area("Write your feedback here", key="feedback_text")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
        logger.info(
            "Feedback received — rating: %s", st.session_state.get("rating")
        )
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
 
    # --- Load resources (crash early with a clear message) ---
    try:
        df    = load_data(DATA_PATH)
        model = load_model(MODEL_PATH)
    except FileNotFoundError as exc:
        st.error(f"Required file not found: {exc.filename}")
        st.stop()
 
    # --- Sidebar ---
    inputs = render_sidebar()
    input_data = encode_inputs(
        inputs["age"], inputs["sex"], inputs["bmi"],
        inputs["children"], inputs["smoker"], inputs["region"],
    )
 
    # --- Hero section ---
    st.title(f"💊 {PAGE_TITLE}")
    st.subheader("Estimate your medical insurance charges")
    st.markdown(
        "This app uses a **Machine Learning** model to estimate insurance costs.  \n"
        "👉 Enter your details in the sidebar, then click **Predict**."
    )
    if IMAGE_PATH.exists():
        st.image(str(IMAGE_PATH), use_container_width=True)
 
    st.divider()
 
    # --- Prediction ---
    if inputs["predict"]:
        render_prediction(model, input_data)
    else:
        st.info("Fill in your information and click **Predict Insurance Cost**.")
 
    st.divider()
 
    # --- EDA ---
    render_eda(df)
 
    st.divider()
 
    # --- Feedback ---
    render_feedback()
 
 
if __name__ == "__main__":
    main()
 