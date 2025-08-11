import os
from pathlib import Path
import base64
import pandas as pd
import streamlit as st

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Loan Default Prediction", layout="wide", page_icon="💸")

# ===================== FILE PATHS =====================
IMAGE_PATH = Path(__file__).parent / "loan.jpg"
DATA_PATH = Path("C:/Users/user/PycharmProjects/Group2SML/data/loan_default.csv")

# ===================== UTILITIES =====================
def get_base64_image(image_path: Path) -> str:
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpg;base64,{encoded}"

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

# ===================== HEADER IMAGE =====================
if IMAGE_PATH.exists():
    image_base64 = get_base64_image(IMAGE_PATH)
    st.markdown(
        f"""
        <div style="width:100%; text-align:center;">
            <img src="{image_base64}" style="width:100%; max-height:250px; object-fit:cover;" alt="Loan Header Image">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("ℹ️ Place a header image named `loan.jpg` in the app folder to show a banner.")

# ===================== INTRO =====================
st.markdown("""
### 📊 Predict Loan Defaults Based on Financial and Demographic Data

Banks rely on lending for revenue, but default risk is real. With historical borrower data, we can build a machine learning model to predict the chance of default for new applicants.

The dataset includes features like income, gender, and loan purpose.
""")

st.markdown("""
Workflow:
- 🔍 Data Exploration
- 📊 Visualization
- 🛠️ Preprocessing
- 🔨 Feature Selection & Scaling
- 🤖 Model Training
- 📉 Evaluation
- 🧮 Interactive Predictions
""")

# ===================== LOAD DATA (FIXED PATH) =====================
st.subheader("Dataset")
if DATA_PATH.exists() and DATA_PATH.is_file():
    try:
        df_default = load_csv(DATA_PATH)
        st.session_state["df_default"] = df_default
        st.success(f"✅ Loaded dataset from: {DATA_PATH}")
        st.caption(f"Rows: {len(df_default):,} | Columns: {len(df_default.columns):,}")
        st.dataframe(df_default.head())

        # Navigation
        st.divider()

    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
else:
    st.error(f"❌ File not found: {DATA_PATH}")
    st.info("Ensure the CSV exists at that location.")
