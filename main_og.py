import pandas as pd
import numpy as np
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load Data
file_path = r".\drug_data.csv"
data = pd.read_csv(file_path)

# Sets
U = list(data['Drug'].unique())
E = ['Age', 'BP', 'Cholesterol', 'Na_to_K']
M = ['p', 'q', 'r']
O = [0, 1]

# Weights
weights = {'e1': 0.15, 'e2': 0.35, 'e3': 0.35, 'e4': 0.15}

# Expert credibility (optional improvement)
expert_credibility = {'p': 0.9, 'q': 0.8, 'r': 0.7}

# Fuzzification Functions
def fuzzify_age(age): return round(np.clip(age / 100, 0.2, 1.0), 2)
def fuzzify_na_to_k(value): return round(np.clip(value / 20, 0.2, 1.0), 2)
def fuzzify_bp(bp): return {'LOW': 0.3, 'NORMAL': 0.8, 'HIGH': 0.4}.get(bp, 0.5)
def fuzzify_cholesterol(ch): return {'NORMAL': 0.8, 'HIGH': 0.2}.get(ch, 0.5)

# IFSES Construction
IFSES = []
for _, row in data.iterrows():
    drug = row['Drug']
    params = (fuzzify_age(row['Age']), fuzzify_bp(row['BP']), 
              fuzzify_cholesterol(row['Cholesterol']), fuzzify_na_to_k(row['Na_to_K']))
    expert = np.random.choice(M)
    opinion = np.random.choice(O)
    IFSES.append((drug, params, expert, opinion))

# Split Agree and Disagree Sets
agree_IFSES = [entry for entry in IFSES if entry[3] == 1]
disagree_IFSES = [entry for entry in IFSES if entry[3] == 0]

# Convert to DataFrame
def to_dataframe(ifses):
    return pd.DataFrame({
        'Drug-Expert': [f"{d},{e}" for d, _, e, _ in ifses],
        'e1': [p[0] for _, p, _, _ in ifses],
        'e2': [p[1] for _, p, _, _ in ifses],
        'e3': [p[2] for _, p, _, _ in ifses],
        'e4': [p[3] for _, p, _, _ in ifses],
    })

agree_df = to_dataframe(agree_IFSES)
disagree_df = to_dataframe(disagree_IFSES)

# Step 1: Save IFSES with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
agree_df.to_excel(f"Agree_IFSES_{timestamp}.xlsx", index=False)
disagree_df.to_excel(f"Disagree_IFSES_{timestamp}.xlsx", index=False)

# Step 2: Normalization
def normalize(df):
    norm_df = df.copy()
    for col in ['e1', 'e2', 'e3', 'e4']:
        norm_df[col] = df[col] / np.sqrt((df[col] ** 2).sum())
    return norm_df

agree_norm = normalize(agree_df)
disagree_norm = normalize(disagree_df)
agree_norm.to_excel(f"Agree_Normalized_{timestamp}.xlsx", index=False)
disagree_norm.to_excel(f"Disagree_Normalized_{timestamp}.xlsx", index=False)

# Step 3: Weighted Aggregation
def weighted_sum(df):
    weight_series = pd.Series(weights, index=['e1', 'e2', 'e3', 'e4'])
    return df[['e1', 'e2', 'e3', 'e4']].dot(weight_series)

agree_norm['Weighted_Sum'] = weighted_sum(agree_norm)
disagree_norm['Weighted_Sum'] = weighted_sum(disagree_norm)
agree_norm.to_excel(f"Agree_Weighted_{timestamp}.xlsx", index=False)
disagree_norm.to_excel(f"Disagree_Weighted_{timestamp}.xlsx", index=False)

# Step 4: Net Score Calculation
net_scores = pd.DataFrame({
    'Drug-Expert': agree_norm['Drug-Expert'],
    'Agree_Sum': agree_norm['Weighted_Sum'],
    'Disagree_Sum': disagree_norm['Weighted_Sum'],
})
net_scores['Net_Score'] = net_scores['Agree_Sum'] - net_scores['Disagree_Sum']
net_scores.sort_values(by='Net_Score', ascending=False, inplace=True)
net_scores.to_excel(f"Drug_Ranking_{timestamp}.xlsx", index=False)

# Determine the best drug
best_drug = net_scores.iloc[0]
print(f"The best drug is: {best_drug['Drug-Expert']} with a net score of {best_drug['Net_Score']}")
