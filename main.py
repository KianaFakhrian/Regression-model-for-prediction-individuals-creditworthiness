import pandas as pd
import numpy as np

df = pd.read_csv('CreditPrediction.csv')

# ============================================================
# 2. Replace empty strings with NaN (treat blanks as missing)
# ============================================================
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.replace('Unknown', np.nan)

# ============================================================
# 3. Define mapping dictionaries (no mapping for "Unknown" or missing)
# ============================================================
gender_map = {'M': 1, 'F': 2}

education_map = {
    'Uneducated': 1,
    'High School': 2,
    'College': 3,
    'Graduate': 4,
    'Post-Graduate': 5,
    'Doctorate': 6
}

marital_map = {
    'Single': 1,
    'Married': 2,
    'Divorced': 3
}

card_map = {
    'Blue': 1,
    'Silver': 2,
    'Gold': 3,
    'Platinum': 4
}

# Income → midpoint (unknown/empty stay NaN)
def income_to_midpoint(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val == 'Less than $40K':
        return 30000
    elif val == '$40K - $60K':
        return 50000
    elif val == '$60K - $80K':
        return 70000
    elif val == '$80K - $120K':
        return 100000
    elif val == '$120K +':
        return 140000
    else:
        return np.nan

# ============================================================
# 4. Apply mappings (unmapped values become NaN automatically)
# ============================================================
df['Gender_Code'] = df['Gender'].map(gender_map)
df['Education_Code'] = df['Education_Level'].map(education_map)
df['Marital_Code'] = df['Marital_Status'].map(marital_map)
df['Card_Code'] = df['Card_Category'].map(card_map)
df['Income_Midpoint'] = df['Income_Category'].apply(income_to_midpoint)

# ============================================================
# 5. Select only the numeric columns for covariance
#    (exclude original categorical columns and CLIENTNUM)
# ============================================================
numeric_cols = [
    'Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Code', 'Education_Code', 'Marital_Code', 'Income_Midpoint', 'Card_Code'
]

# Make a copy with only these columns
df_numeric = df[numeric_cols].copy()

# ============================================================
# 6. Compute the covariance matrix
#    (pandas .cov() automatically handles NaN using pairwise deletion)
# ============================================================
cov_matrix = df_numeric.cov()

print("Covariance Matrix (missing values treated as NaN):\n")
print(cov_matrix)

# ============================================================
# 7. (Optional) Save to CSV for use in Word/Excel
# ============================================================
cov_matrix.to_csv('covariance_matrix_with_missing.csv')
print("\nCovariance matrix saved to 'covariance_matrix_with_missing.csv'")

# ============================================================
# 8. (Optional) Check how many missing values remain
# ============================================================
print("\nMissing value counts per column:\n")
print(df_numeric.isna().sum())