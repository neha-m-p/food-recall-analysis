# 🥫 FDA Food Recall Analysis

This project analyzes FDA food recall data to uncover major recall causes, high-risk companies, affected states, and long-term trends.  
The analysis explores patterns in contamination types, product issues, and recall frequency over time using Python, pandas, and Prophet.

---

## 📌 Project Objectives

The goal of this project is to:

- Identify the **most common recall causes**
- Analyze **high-risk companies** and **high-risk product types**
- Determine **states most affected** by recalls
- Study **trend patterns** across months and years
- Examine whether recalls are **increasing or decreasing**
- Categorize recall reasons into meaningful groups

This project helps improve **food safety decisions**, **regulatory planning**, and **risk prevention strategies**.

---

## 📁 Dataset

The dataset used:


recalls-market-withdrawals-safety-alerts.xlsx


It contains FDA recall notices including:

- Recall date  
- Product description  
- Recall reason  
- Recalling firm  
- State  
- Recall class / category  

The script loads the dataset using:

```python
df = pd.read_excel("recalls-market-withdrawals-safety-alerts.xlsx")
🛠️ Technologies Used
Purpose	Libraries
Data Processing	pandas, numpy
Visualization	matplotlib, seaborn
Forecasting	Prophet
Warning Control	warnings
🔧 Features Implemented in Code
✔ 1. Data Loading
Reads Excel file
Displays dataset size
Shows sample rows
✔ 2. Data Cleaning
Removes duplicate rows
Converts Date column to datetime
Extracts:
Year
Month
Month_Year
Fills missing text fields with "Unknown"
Cleans text columns (lowercase, strip spaces)
✔ 3. Recall Reason Processing

Creates a cleaned text version:

df["Reason_Clean"]

Then categorizes each recall into:

Biological Contamination
Allergen Issue
Mislabeling
Foreign Material
General Contamination
Other
✔ 4. Additional Feature Engineering

Adds:

Recall_Category → grouped category
Desc_Length → length of product description
📊 Insights You Can Generate

Using this codebase, you can analyze:

Recall Trends
Month-wise and year-wise recall counts
Seasonality patterns
Root Causes
Top reasons: allergens, contamination, mislabeling, etc.
Geographical Trends
States with most recall reports
Company-Level Insights
Firms with frequent recalls
Forecasting
Predict future recall frequency using Prophet
▶️ How to Run This Project
1. Install required libraries
pip install pandas numpy matplotlib seaborn prophet
2. Place the Excel dataset in the same folder
/FDA-Food-Recall-Analysis
    |
    ├── analysis_script.py
    ├── recalls-market-withdrawals-safety-alerts.xlsx
    └── README.md
3. Run the analysis script
python analysis_script.py
📌 Project Folder Structure
FDA-Food-Recall-Analysis/
│
├── analysis_script.py              # Main analysis code
├── recalls-market-withdrawals.xlsx # Dataset
├── README.md                       # This file
└── outputs/                        # Generated charts, processed data (optional)
🌟 Future Improvements
Build an interactive dashboard (Streamlit or Power BI)
Add product-level risk scoring
Integrate geospatial maps
Implement automated EDA reports
