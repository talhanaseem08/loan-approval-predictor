
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

# Required columns for the app (must be present in uploaded CSV)
TARGET = "approved"
REQUIRED_COLS = [
    "gender", "age", "city", "employment_type", "bank",
    "monthly_income_pkr", "credit_score", "loan_amount_pkr",
    "loan_tenure_months", "existing_loans", "default_history",
    "has_credit_card", TARGET,
]

# --- Sidebar: Upload CSV & cache ---
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

with st.sidebar:
    st.header("📁 Dataset")
    uploaded_file = st.file_uploader(
        "Upload a loan CSV",
        type=["csv"],
        help="Upload a CSV with the same columns as the default dataset (e.g. approved, gender, age, city, ...). It will be cached for this session.",
    )
    if uploaded_file is not None:
        try:
            up_df = pd.read_csv(uploaded_file)
            missing = [c for c in REQUIRED_COLS if c not in up_df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}. Using default dataset.")
            elif len(up_df) < 100:
                st.warning("Dataset has very few rows. Using default dataset.")
            else:
                st.session_state.uploaded_df = up_df
                st.session_state.uploaded_filename = uploaded_file.name
                st.success(f"Cached: **{uploaded_file.name}** ({len(up_df):,} rows)")
        except Exception as e:
            st.error(f"Could not read CSV: {e}. Using default dataset.")
    if st.button("Use default dataset"):
        st.session_state.uploaded_df = None
        st.session_state.uploaded_filename = None
        st.rerun()
    if st.session_state.uploaded_filename:
        st.caption(f"Current: **{st.session_state.uploaded_filename}** (cached)")

# --- Load dataset: cached upload or default file ---
@st.cache_data
def load_default_data():
    import os
    path = "loan_dataset_cleaned.csv" if os.path.exists("loan_dataset_cleaned.csv") else "loan_dataset.csv"
    return pd.read_csv(path)

if st.session_state.uploaded_df is not None:
    df = st.session_state.uploaded_df.copy()
else:
    df = load_default_data()

st.title("Loan Approval Predictor")
st.markdown("ML pipeline with Logistic Regression")

# Feature columns (exclude target and any ID column like applicant_name)
feature_cols = [c for c in df.columns if c != TARGET and c != "applicant_name"]

NUMERIC_FEATURES = [
    "age",
    "monthly_income_pkr",
    "credit_score",
    "loan_amount_pkr",
    "loan_tenure_months",
    "existing_loans",
    "default_history",
    "has_credit_card",
]
CATEGORICAL_FEATURES = ["gender", "city", "employment_type", "bank"]

X = df[feature_cols]
y = df[TARGET]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing Pipeline ---
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ]
)

# Full pipeline: preprocess + Logistic Regression
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

# --- Train model ---
with st.spinner("Training model..."):
    model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

# --- Tabs: Metrics & Confusion Matrix | Predict ---
tab1, tab2, tab3 = st.tabs(["📊 Metrics & Confusion Matrix", "🔮 Predict Approval", "📁 Dataset Info"])

with tab1:
    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    with col2:
        st.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
    with col3:
        st.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
    with col4:
        st.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")

    st.subheader("Classification Report")
    # Build report as dict and show as table with proper headings
    report = classification_report(
        y_test, y_pred, target_names=["Not Approved", "Approved"], output_dict=True
    )
    # Rows for each class + accuracy (report dict has 'accuracy' as key, rest are class names)
    rows = []
    for name in ["Not Approved", "Approved"]:
        rows.append({
            "Class": name,
            "Precision": f"{report[name]['precision']:.2f}",
            "Recall": f"{report[name]['recall']:.2f}",
            "F1-Score": f"{report[name]['f1-score']:.2f}",
            "Support": int(report[name]["support"]),
        })
    rows.append({
        "Class": "Accuracy",
        "Precision": "—",
        "Recall": "—",
        "F1-Score": f"{report['accuracy']:.2f}",
        "Support": len(y_test),
    })
    for name in ["macro avg", "weighted avg"]:
        rows.append({
            "Class": name.title(),
            "Precision": f"{report[name]['precision']:.2f}",
            "Recall": f"{report[name]['recall']:.2f}",
            "F1-Score": f"{report[name]['f1-score']:.2f}",
            "Support": int(report[name]["support"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("What do these 4 values mean?"):
        st.markdown("""
        | **Metric** | **Meaning** |
        |------------|-------------|
        | **Precision** | Of all samples the model predicted as this class, how many were correct? *Formula:* True Positives ÷ (True Positives + False Positives) |
        | **Recall** | Of all actual samples of this class, how many did the model find? *Formula:* True Positives ÷ (True Positives + False Negatives) |
        | **F1-Score** | Balance between precision and recall. *Formula:* 2 × (Precision × Recall) ÷ (Precision + Recall) |
        | **Support** | Number of actual samples of this class in the test set. |
        """)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Approved", "Approved"],
        yticklabels=["Not Approved", "Approved"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close()

with tab2:
    st.subheader("Enter applicant details to predict loan approval")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", options=sorted(df["gender"].unique()))
            city = st.selectbox("City", options=sorted(df["city"].unique()))
            employment_type = st.selectbox(
                "Employment Type", options=sorted(df["employment_type"].unique())
            )
            bank = st.selectbox("Bank", options=sorted(df["bank"].unique()))
        with c2:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            monthly_income_pkr = st.number_input(
                "Monthly Income (PKR)", min_value=0, value=100000, step=5000
            )
            credit_score = st.slider("Credit Score", 300, 850, 650)
            loan_amount_pkr = st.number_input(
                "Loan Amount (PKR)", min_value=0, value=500000, step=10000
            )
            loan_tenure_months = st.selectbox(
                "Loan Tenure (months)", options=[6, 12, 18, 24, 36, 48, 60]
            )
            existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)
            default_history = st.selectbox("Default History (ever defaulted?)", [0, 1])
            has_credit_card = st.selectbox("Has Credit Card?", [0, 1])

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "age": age,
                    "city": city,
                    "employment_type": employment_type,
                    "bank": bank,
                    "monthly_income_pkr": monthly_income_pkr,
                    "credit_score": credit_score,
                    "loan_amount_pkr": loan_amount_pkr,
                    "loan_tenure_months": loan_tenure_months,
                    "existing_loans": existing_loans,
                    "default_history": default_history,
                    "has_credit_card": has_credit_card,
                }
            ]
        )
        pred = model_pipeline.predict(input_df)[0]
        proba = model_pipeline.predict_proba(input_df)[0]
        if pred == 1:
            st.success(f"✅ **Approved** — Probability of approval: {proba[1]:.2%}")
        else:
            st.error(f"❌ **Not Approved** — Probability of approval: {proba[1]:.2%}")

with tab3:
    st.subheader("Dataset overview")
    st.write(f"Rows: **{len(df):,}** | Columns: **{len(df.columns)}**")
    st.dataframe(df.head(100), use_container_width=True)


