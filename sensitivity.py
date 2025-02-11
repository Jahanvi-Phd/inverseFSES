import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import matplotlib.pyplot as plt

#Current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = "test_output"
os.makedirs(output_folder, exist_ok=True)
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
agree_df.to_excel(f"{output_folder}/Agree_IFSES_{timestamp}.xlsx", index=False)
disagree_df.to_excel(f"{output_folder}/Disagree_IFSES_{timestamp}.xlsx", index=False)

# Step 2: Normalization
# def old_normalize(df):
#     norm_df = df.copy()
#     print(norm_df)
#     for col in ['e1', 'e2', 'e3', 'e4']:
#         norm_df[col] = df[col] / np.sqrt((df[col] ** 2).sum())
#     return norm_df

def row_normalize(df):
    norm_df = df.copy()
    for index, row in df.iterrows():
        row_sum_squares = np.sqrt((row[['e1', 'e2', 'e3', 'e4']] ** 2).sum())
        norm_df.loc[index, ['e1', 'e2', 'e3', 'e4']] = row[['e1', 'e2', 'e3', 'e4']] / row_sum_squares
    return norm_df

agree_norm = row_normalize(agree_df)
disagree_norm = row_normalize(disagree_df)
agree_norm.to_excel(f"{output_folder}/Agree_Normalized_{timestamp}.xlsx", index=False)
disagree_norm.to_excel(f"{output_folder}/Disagree_Normalized_{timestamp}.xlsx", index=False)

# Step 3: Weighted Aggregation
def weighted_sum(df):
    weight_series = pd.Series(weights, index=['e1', 'e2', 'e3', 'e4'])
    return df[['e1', 'e2', 'e3', 'e4']].dot(weight_series)

agree_norm['Weighted_Sum'] = weighted_sum(agree_norm)
disagree_norm['Weighted_Sum'] = weighted_sum(disagree_norm)
agree_norm.to_excel(f"{output_folder}/Agree_Weighted_{timestamp}.xlsx", index=False)
disagree_norm.to_excel(f"{output_folder}/Disagree_Weighted_{timestamp}.xlsx", index=False)

# Step 4: Net Score Calculation
net_scores = pd.DataFrame({
    'Drug-Expert': agree_norm['Drug-Expert'],
    'Agree_Sum': agree_norm['Weighted_Sum'],
    'Disagree_Sum': disagree_norm['Weighted_Sum'],
})
net_scores['Net_Score'] = net_scores['Agree_Sum'] - net_scores['Disagree_Sum']
net_scores.sort_values(by='Net_Score', ascending=False, inplace=True)
net_scores.to_excel(f"{output_folder}/Drug_Ranking_{timestamp}.xlsx", index=False)

# Determine the best drug
best_drug = net_scores.iloc[0]
print(f"The best drug is: {best_drug['Drug-Expert']} with a net score of {best_drug['Net_Score']}")

# Create a figure with multiple subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Sensitivity Analysis for Different Weight Combinations', fontsize=16)

# Weight variations
weight_variations = {
    'Original': {'e1': 0.15, 'e2': 0.35, 'e3': 0.35, 'e4': 0.15},
    'Equal': {'e1': 0.25, 'e2': 0.25, 'e3': 0.25, 'e4': 0.25},
    'Age-Focus': {'e1': 0.40, 'e2': 0.20, 'e3': 0.20, 'e4': 0.20},
    'BP-Focus': {'e1': 0.20, 'e2': 0.40, 'e3': 0.20, 'e4': 0.20}
}

def plot_sensitivity(agree_norm, disagree_norm, weights, ax, title):
    # Calculate weighted sums
    agree_weighted = agree_norm[['e1', 'e2', 'e3', 'e4']].dot(pd.Series(weights))
    disagree_weighted = disagree_norm[['e1', 'e2', 'e3', 'e4']].dot(pd.Series(weights))
    
    # Calculate net scores
    net_scores = agree_weighted - disagree_weighted
    sorted_scores = sorted(net_scores, reverse=True)
    
    # Plot
    ax.plot(range(len(sorted_scores)), sorted_scores, marker='o')
    ax.set_title(title)
    ax.set_xlabel('Drug-Expert Combinations')
    ax.set_ylabel('Net Score')
    ax.grid(True)

# Plot each variation
plot_sensitivity(agree_norm, disagree_norm, weight_variations['Original'], ax1, 'Original Weights')
plot_sensitivity(agree_norm, disagree_norm, weight_variations['Equal'], ax2, 'Equal Weights')
plot_sensitivity(agree_norm, disagree_norm, weight_variations['Age-Focus'], ax3, 'Age-Focused Weights')
plot_sensitivity(agree_norm, disagree_norm, weight_variations['BP-Focus'], ax4, 'BP-Focused Weights')

# Adjust layout
plt.tight_layout()
plt.savefig(f'sensitivity_analysis_combined{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()