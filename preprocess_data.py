"""
Clean and preprocess loan_dataset.csv:
- Remove unnecessary columns (e.g. applicant_name - not used in ML)
- Handle missing values
- Remove duplicates
- Strip whitespace in categorical columns
- Validate numeric ranges (drop invalid rows)
- Save as loan_dataset_cleaned.csv
"""

import pandas as pd

# Load
df = pd.read_csv("loan_dataset.csv")
print(f"Original: {len(df)} rows, {len(df.columns)} columns")
print("Columns:", list(df.columns))

# 1. Drop unnecessary columns (identifier, not a feature for prediction)
df = df.drop(columns=["applicant_name"], errors="ignore")
print(f"\nDropped 'applicant_name'. Columns now: {list(df.columns)}")

# 2. Strip whitespace from string columns
str_cols = df.select_dtypes(include=["object"]).columns
for c in str_cols:
    df[c] = df[c].astype(str).str.strip()

# 3. Missing values
missing = df.isnull().sum()
if missing.any():
    print(f"\nMissing values before:\n{missing[missing > 0]}")
    df = df.dropna()
    print(f"After dropna: {len(df)} rows")
else:
    print("\nNo missing values.")

# 4. Remove duplicate rows (all columns)
before = len(df)
df = df.drop_duplicates()
print(f"\nDuplicates removed: {before - len(df)}. Rows now: {len(df)}")

# 5. Sanity checks on numeric columns (drop invalid rows)
# Age: reasonable range
df = df[(df["age"] >= 18) & (df["age"] <= 100)]
# Income and loan amount: non-negative
df = df[(df["monthly_income_pkr"] >= 0) & (df["loan_amount_pkr"] >= 0)]
# Credit score: typical range
df = df[(df["credit_score"] >= 300) & (df["credit_score"] <= 850)]
# Tenure: positive
df = df[df["loan_tenure_months"] > 0]
# Binary flags: 0 or 1
df = df[(df["default_history"].isin([0, 1])) & (df["has_credit_card"].isin([0, 1])) & (df["approved"].isin([0, 1]))]
df = df[df["existing_loans"] >= 0]

print(f"After numeric validation: {len(df)} rows")

# 6. Save cleaned dataset
df.to_csv("loan_dataset_cleaned.csv", index=False)
print(f"\nSaved: loan_dataset_cleaned.csv ({len(df)} rows, {len(df.columns)} columns)")
