import base64
from pathlib import Path
import pandas as pd
import streamlit as st

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Loan Default Prediction", layout="wide", page_icon="💸")

# ===================== FILE PATHS =====================
BASE = Path(__file__).parent
IMAGE_PATH = BASE / "loan.jpg"
DATA_PATH = BASE / "data" / "Loan_Default.csv"

# ===================== UTILITIES =====================
def get_base64_image(image_path: Path) -> str:
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpg;base64,{encoded}"

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Robust CSV loader: infers delimiter and handles BOM."""
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, low_memory=False)
        except Exception:
            continue
    # last attempt with defaults to surface a clear error
    return pd.read_csv(path, low_memory=False)

# ===================== HEADER IMAGE =====================
if IMAGE_PATH.exists():
    try:
        image_base64 = get_base64_image(IMAGE_PATH)
        st.markdown(
            f"""
            <div style="width:100%; text-align:center;">
                <img src="{image_base64}" style="width:100%; max-height:250px; object-fit:cover;" alt="Loan Header Image">
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        # Fallback if base64 fails for any reason
        st.image(str(IMAGE_PATH), use_container_width=True, caption=f"(Rendered without base64: {e})")
else:
    st.info("Place a header image named `loan.jpg` in the app folder to show a banner.")

# ===================== INTRO =====================
st.markdown("""
### 📊 Predict Loan Defaults Based on Financial and Demographic Data

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
        if df_default is None or df_default.empty:
            st.error("The CSV loaded but appears to be empty.")
        else:
            # save for other pages
            st.session_state["df_default"] = df_default
            st.session_state["df_active"] = df_default
            st.success(f"✅ Loaded dataset from: {DATA_PATH}")
            st.caption(f"Rows: {len(df_default):,} | Columns: {df_default.shape[1]:,}")
            st.dataframe(df_default.head(50), use_container_width=True)
            st.divider()
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
else:
    st.error(f"❌ File not found: {DATA_PATH}")
    st.info("Ensure the CSV exists at repo_root/data/loan_default.csv.")
