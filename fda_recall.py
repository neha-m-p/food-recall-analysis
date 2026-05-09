"""
==========================================================
                FDA FOOD RECALL ANALYSIS
==========================================================

📌 OBJECTIVE
Analyze historical FDA food recall data to identify:
- Major reasons for recalls
- High-risk companies, products, and states
- Trend patterns over time
- Key insights that help prevent future recalls

📌 WHY THIS PROJECT MATTERS
Food recalls create:
- Health risks
- Brand damage
- Financial losses
- Regulatory impact

Understanding recall patterns helps manufacturers 
and regulators improve food safety.

📌 KEY QUESTIONS
1. What are the most common recall causes?
2. Which companies are responsible for most recalls?
3. Which states are most affected?
4. Which product types are high-risk?
5. Which months/years have peak recalls?
6. Are recalls increasing or decreasing over time?
==========================================================
"""

# ==========================================================
# STEP 2: IMPORT LIBRARIES & LOAD DATA
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from prophet import Prophet
warnings.filterwarnings("ignore")

# ---------- Load Dataset ----------
file_path = "recalls-market-withdrawals-safety-alerts.xlsx"

df = pd.read_excel(file_path)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
print("Dataset Loaded Successfully!")
print("Rows:", df.shape[0], " Columns:", df.shape[1])
print(df.head())

# ==========================================================
# STEP 3: DATA CLEANING & PREPROCESSING
# ==========================================================

print("\n---- Cleaning Data ----")

# 1. Remove duplicate rows
df.drop_duplicates(inplace=True)


# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Extract Year and Month
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month_Year"] = df["Date"].dt.to_period("M").astype(str)
print(df.head)
# print("✓ Date converted. Created: Year, Month, Month_Year")
# 4. Replace blank/NaN text fields with "Unknown"
text_columns = ["Product Description", "Reason for Recall", "Recalling Firm", "State"]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

# 5. Clean Reason column (remove punctuation, lowercase)
if "Reason for Recall" in df.columns:
    df["Reason_Clean"] = (
        df["Reason for Recall"]
        .str.replace("[^a-zA-Z ]", "", regex=True)
        .str.lower()
    )

print("Data Cleaning Completed.\n")
print(df.info())
print(df.columns)

print("\n---- FIX: Creating Year, Month, and Month_Year Columns ----")

print(df[["Date", "Year", "Month", "Month_Year"]].head())

print("\n---- Step 4: Creating New Columns (Recall Categories, Description Length) ----")

# 1. Clean recall reason
df["Reason_Clean"] = (
    df["Recall-Reason-Description"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# 2. Categorize recall reason into groups
def categorize_reason(text):
    text = str(text).lower()

    if "salmonella" in text or "listeria" in text or "e.coli" in text:
        return "Biological Contamination"

    elif "undeclared" in text or "allergen" in text:
        return "Allergen Issue"

    elif "mislabel" in text or "wrong label" in text or "false" in text:
        return "Mislabeling"

    elif "foreign" in text or "plastic" in text or "metal" in text:
        return "Foreign Material"

    elif "contamination" in text:
        return "General Contamination"

    else:
        return "Other"

df["Recall_Category"] = df["Reason_Clean"].apply(categorize_reason)

# 3. Description length
df["Desc_Length"] = df["Product-Description"].apply(
    lambda x: len(str(x).split())
)

print("✓ New columns created: Reason_Clean, Recall_Category, Desc_Length")


print("\n---- Step 5: Exploratory Data Analysis (EDA) ----")


# ---------------- 1. Recall Count per Year ----------------
plt.figure(figsize=(8,4))
sns.countplot(x=df["Year"], palette="pastel")
plt.title("Number of Recalls per Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("recalls_per_year.png")
plt.close()

# ---------------- 2. Recall Categories Distribution ----------------
plt.figure(figsize=(8,4))
df["Recall_Category"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Distribution of Recall Categories")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("recall_categories.png")
plt.close()

# ---------------- 3. Top 10 Companies with Most Recalls ----------------
top_companies = df["Company-Name"].value_counts().head(10)

plt.figure(figsize=(8,4))
sns.barplot(x=top_companies.values, y=top_companies.index, palette="viridis")
plt.title("Top 10 Companies with the Most Recalls")
plt.tight_layout()
plt.savefig("top_companies.png")
plt.close()

# ---------------- 4. Product Type Distribution ----------------
plt.figure(figsize=(8,4))
df["Product-Types"].value_counts().head(10).plot(kind="bar", color="salmon")
plt.title("Top Product Types Recalled")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("product_types.png")
plt.close()

# ---------------- 5. Description Length Distribution ----------------
plt.figure(figsize=(8,4))
sns.histplot(df["Desc_Length"], bins=20, kde=True)
plt.title("Distribution of Product Description Length")
plt.xlabel("Number of Words")
plt.tight_layout()
plt.savefig("description_length_distribution.png")
plt.close()

print("✓ EDA completed! Charts saved: ")
print(" - recalls_per_year.png")
print(" - recall_categories.png")
print(" - top_companies.png")
print(" - product_types.png")
print(" - description_length_distribution.png")

print("\n---- Step 6: Trend Analysis + High-Risk Company & Product Detection ----")

# -----------------------------------------------------
# 1. Monthly Recall Trend (line chart)
# -----------------------------------------------------
monthly_trend = df.groupby("Month_Year").size()

plt.figure(figsize=(12,5))
monthly_trend.plot(kind="line", marker="o")
plt.title("Monthly Recall Trend Over Time")
plt.xlabel("Month_Year")
plt.ylabel("Number of Recalls")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_recall_trend.png")
plt.close()


# -----------------------------------------------------
# 2. Yearly Recall Trend (bar chart)
# -----------------------------------------------------
yearly_trend = df.groupby("Year").size()

plt.figure(figsize=(8,4))
yearly_trend.plot(kind="bar", color="teal")
plt.title("Total Recalls Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Recalls")
plt.tight_layout()
plt.savefig("yearly_recall_trend.png")
plt.close()


# -----------------------------------------------------
# 3. High-Risk Companies (Top 10)
# -----------------------------------------------------
high_risk_companies = df["Company-Name"].value_counts().head(10)

plt.figure(figsize=(8,4))
sns.barplot(x=high_risk_companies.values, y=high_risk_companies.index, palette="crest")
plt.title("Top 10 High-Risk Companies (Most Recalls)")
plt.xlabel("Number of Recalls")
plt.tight_layout()
plt.savefig("high_risk_companies.png")
plt.close()


# -----------------------------------------------------
# 4. High-Risk Product Types
# -----------------------------------------------------
high_risk_products = df["Product-Types"].value_counts().head(10)

plt.figure(figsize=(8,4))
sns.barplot(x=high_risk_products.values, y=high_risk_products.index, palette="flare")
plt.title("Top 10 High-Risk Product Types")
plt.xlabel("Number of Recalls")
plt.tight_layout()
plt.savefig("high_risk_product_types.png")
plt.close()


# -----------------------------------------------------
# 5. Seasonal Pattern (Which Months Have Most Recalls?)
# -----------------------------------------------------
monthly_pattern = df["Month"].value_counts().sort_index()

plt.figure(figsize=(8,4))
sns.barplot(x=monthly_pattern.index, y=monthly_pattern.values, palette="pastel")
plt.title("Which Months Have the Most Recalls?")
plt.xlabel("Month")
plt.ylabel("Recall Count")
plt.tight_layout()
plt.savefig("monthly_seasonal_pattern.png")
plt.close()


print("✓ Step 6 completed successfully! Charts saved.")

print("\n---- Step 7: Root Cause Pattern Mining (N-Gram Text Analysis) ----")

import re
from sklearn.feature_extraction.text import CountVectorizer

# Use the cleaned reason column from Step 4
text_data = df["Reason_Clean"].dropna().astype(str)

# ---------------------------------------------------------
# Function to extract and plot N-grams
# ---------------------------------------------------------
def plot_top_ngrams(text, n=1, top_k=20, file_name="ngrams.png"):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(text)
    bag = vec.transform(text)
    sum_words = bag.sum(axis=0)

    words_freq = [
        (word, sum_words[0, idx]) 
        for word, idx in vec.vocabulary_.items()
    ]

    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]

    # Plot
    plt.figure(figsize=(10,5))
    sns.barplot(x=[x[1] for x in words_freq], y=[x[0] for x in words_freq], palette="coolwarm")
    plt.title(f"Top {top_k} {n}-grams in Recall Reasons")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


# ---------------------------------------------------------
# Run for Unigrams (1-word patterns)
# ---------------------------------------------------------
plot_top_ngrams(text_data, n=1, top_k=20, file_name="unigrams.png")

# ---------------------------------------------------------
# Run for Bigrams (2-word patterns)
# ---------------------------------------------------------
plot_top_ngrams(text_data, n=2, top_k=20, file_name="bigrams.png")

# ---------------------------------------------------------
# Run for Trigrams (3-word patterns)
# ---------------------------------------------------------
plot_top_ngrams(text_data, n=3, top_k=20, file_name="trigrams.png")

print("✓ Step 8 completed! N-gram charts saved: unigrams.png, bigrams.png, trigrams.png")

# ---- STEP 8: TIME-SERIES FORECASTING USING PROPHET ----



# 1. Aggregate recalls per month
monthly_recalls = df.groupby("Month_Year").size().reset_index(name="Count")

# Convert Month_Year to proper date format (first day of month)
monthly_recalls["Month_Year"] = pd.to_datetime(monthly_recalls["Month_Year"], format="%Y-%m")

# Prophet needs columns renamed as ds and y
forecast_df = monthly_recalls.rename(columns={"Month_Year": "ds", "Count": "y"})

# 2. Create the Prophet model
model = Prophet()
model.fit(forecast_df)

# 3. Create future dataframe (predict next 6 months)
future = model.make_future_dataframe(periods=6, freq="M")

# 4. Forecast
forecast = model.predict(future)

# 5. Plot forecast
model.plot(forecast)
plt.title("FDA Recall Forecast (Next 6 Months)")
plt.xlabel("Date")
plt.ylabel("Predicted Recall Count")
plt.tight_layout()
plt.savefig("recall_forecast.png")
plt.show()

# Trend components
model.plot_components(forecast)
plt.tight_layout()
plt.savefig("recall_forecast_components.png")
plt.show()

print("✓ Step 8 completed successfully! Forecast charts saved:")
print(" - recall_forecast.png")
print(" - recall_forecast_components.png")

print("\n---- Step 9: Machine Learning Classification Model ----")

# -----------------------------
# 1. CLEAN THE TARGET COLUMN
# -----------------------------
if "Terminated Recall" not in df.columns:
    raise ValueError("ERROR: 'Terminated Recall' column not found.")

print("\nOriginal label distribution (raw):")
print(df["Terminated Recall"].value_counts(dropna=False))


# Replace missing values with a meaningful label
df["Terminated Recall"] = df["Terminated Recall"].fillna("Not Terminated")

# Normalize text values
df["Terminated Recall"] = df["Terminated Recall"].astype(str).str.strip().str.lower()

# Map to 0/1
label_map = {
    "terminated": 1,
    "yes": 1,
    "y": 1,
    "1": 1,
    "not terminated": 0,
    "no": 0,
    "n": 0,
    "0": 0,
    "ongoing": 0,
    "on-going": 0,
    "open": 0,
    "active": 0,
}

df["Recall_Label"] = df["Terminated Recall"].map(label_map)

print("\nCleaned label distribution:")
print(df["Recall_Label"].value_counts())


# -------------------------------------------
# 2. ENSURE DATA HAS BOTH CLASSES (0 and 1)
# -------------------------------------------
if df["Recall_Label"].nunique() < 2:
    print("\n❌ ERROR: Only one class found after cleaning. Cannot train model.")
    print("Model skipped due to insufficient label variation.")
else:
    # -------------------------------------------
    # 3. SELECT FEATURES FOR MODEL
    # -------------------------------------------
    features = [
        "Desc_Length",
        "Year",
        "Month",
    ]

    # Confirm feature columns exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"ERROR: Missing feature columns: {missing}")

    X = df[features]
    y = df["Recall_Label"]

    # -------------------------------
    # 4. Train-test split
    # -------------------------------
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------
    # 5. Train model
    # -------------------------------
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # -------------------------------
    # 6. Predictions
    # -------------------------------
    y_pred = model.predict(X_test)

    # -------------------------------
    # 7. Evaluation
    # -------------------------------
    from sklearn.metrics import classification_report

    print("\n---- Classification Report ----")
    print(classification_report(y_test, y_pred))

    print("✓ Step 9 completed successfully!")