# Assignment 09 — Loan Approval Streamlit App

ML pipeline: load dataset → preprocess (Pipeline) → train Logistic Regression → show metrics & confusion matrix → predict from user input.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud (to get your **Live Link**)

1. Push this folder to a **GitHub** repository (e.g. `your-username/loan-approval-app`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**.
4. Set:
   - **Repository**: `your-username/loan-approval-app`
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `app.py`
5. Click **Deploy**. Wait for the build to finish.
6. Your **live link** will be: `https://your-app-name.streamlit.app`

**Important:** `loan_dataset.csv` must be in the same repo and same folder as `app.py` so the app can load it.

## What the app does

- Loads `loan_dataset.csv`
- Preprocesses with a **Pipeline** (ColumnTransformer: StandardScaler + OneHotEncoder)
- Trains **Logistic Regression**
- Shows **metrics** (accuracy, precision, recall, F1) and **confusion matrix**
- **Predict** tab: user inputs → approval prediction with probability
