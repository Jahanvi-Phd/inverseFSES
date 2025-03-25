import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
from collections import Counter

#Current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"test_output/output_{timestamp}"
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
#setting equal number o f0 and 1 i.e 100 each
opinions = [0] * 100 + [1] * 100
np.random.shuffle(opinions)


for i, row in data.iterrows():
    drug = row['Drug']
    params = (fuzzify_age(row['Age']), fuzzify_bp(row['BP']), 
              fuzzify_cholesterol(row['Cholesterol']), fuzzify_na_to_k(row['Na_to_K']))
    expert = np.random.choice(M)
    opinion = opinions[i]
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

# Normalization
def row_normalize(df):
    norm_df = df.copy()
    for index, row in df.iterrows():
        row_sum_squares = np.sqrt((row[['e1', 'e2', 'e3', 'e4']] ** 2).sum())
        norm_df.loc[index, ['e1', 'e2', 'e3', 'e4']] = row[['e1', 'e2', 'e3', 'e4']] / row_sum_squares
    return norm_df

# Step 3: Weighted Aggregation
def weighted_sum(df):
    weight_series = pd.Series(weights, index=['e1', 'e2', 'e3', 'e4'])
    return df[['e1', 'e2', 'e3', 'e4']].dot(weight_series)

agree_normalised_weighted = row_normalize(agree_df)
disagree_normalised_weighted = row_normalize(disagree_df)
agree_normalised_weighted['Weighted_Sum'] = weighted_sum(agree_normalised_weighted)
disagree_normalised_weighted['Weighted_Sum'] = weighted_sum(disagree_normalised_weighted)

# Step 4: Net Score Calculation
net_scores = pd.DataFrame({
    'Drug-Expert': agree_normalised_weighted['Drug-Expert'],
    'Agree_Sum': agree_normalised_weighted['Weighted_Sum'],
    'Disagree_Sum': disagree_normalised_weighted['Weighted_Sum'],
})
net_scores['Net_Score'] = net_scores['Agree_Sum'] - net_scores['Disagree_Sum']
net_scores.sort_values(by='Net_Score', ascending=False, inplace=True)

# Determine the best drug
best_drug = net_scores.iloc[0]
print(f"The best drug is: {best_drug['Drug-Expert']} with a net score of {best_drug['Net_Score']}")


# Weight variations
weight_variations = {
    'Original': {'e1': 0.15, 'e2': 0.35, 'e3': 0.35, 'e4': 0.15},
    'Equal': {'e1': 0.25, 'e2': 0.25, 'e3': 0.25, 'e4': 0.25},
    'Age-Focus': {'e1': 0.40, 'e2': 0.20, 'e3': 0.20, 'e4': 0.20},
    'BP-Focus': {'e1': 0.20, 'e2': 0.40, 'e3': 0.20, 'e4': 0.20}
}

def plot_sensitivity(agree_normalised_weighted, disagree_normalised_weighted, weights, ax, title):
    # Calculate weighted sums
    agree_weighted = agree_normalised_weighted[['e1', 'e2', 'e3', 'e4']].dot(pd.Series(weights))
    disagree_weighted = disagree_normalised_weighted[['e1', 'e2', 'e3', 'e4']].dot(pd.Series(weights))
    
    # Calculate net scores
    net_scores = agree_weighted - disagree_weighted
    sorted_scores = sorted(net_scores, reverse=True)
    
    # Plot
    ax.plot(range(len(sorted_scores)), sorted_scores, marker='o')
    ax.set_title(title)
    ax.set_xlabel('Drug-Expert Combinations')
    ax.set_ylabel('Net Score')
    ax.grid(True)

# Create a figure with multiple subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Sensitivity Analysis for Different Weight Combinations', fontsize=15)
# Plot each variation
plot_sensitivity(agree_normalised_weighted, disagree_normalised_weighted, weight_variations['Original'], ax1, 'Original Weights')
plot_sensitivity(agree_normalised_weighted, disagree_normalised_weighted, weight_variations['Equal'], ax2, 'Equal Weights')
plot_sensitivity(agree_normalised_weighted, disagree_normalised_weighted, weight_variations['Age-Focus'], ax3, 'Age-Focused Weights')
plot_sensitivity(agree_normalised_weighted, disagree_normalised_weighted, weight_variations['BP-Focus'], ax4, 'BP-Focused Weights')

# Save all DataFrames to a single Excel file
with pd.ExcelWriter(f"{output_folder}/IFSES_Output_{timestamp}.xlsx") as writer:
    agree_df.to_excel(writer, sheet_name='Agree_IFSES', index=False)
    disagree_df.to_excel(writer, sheet_name='Disagree_IFSES', index=False)
    agree_normalised_weighted.to_excel(writer, sheet_name='Agree_Weighted', index=False)
    disagree_normalised_weighted.to_excel(writer, sheet_name='Disagree_Weighted', index=False)
    net_scores.to_excel(writer, sheet_name='Drug_Ranking', index=False)

# Adjust layout
plt.tight_layout()
plt.savefig(f'{output_folder}/sensitivity_analysis_combined_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# Add this function to plot all variations together
def plot_combined_sensitivity(agree_normalised_weighted, disagree_normalised_weighted, weight_variations, ax):
    colors = ['b', 'g', 'r', 'c']
    labels = ['Original', 'Equal', 'Age-Focus', 'BP-Focus']
    for i, (key, weights) in enumerate(weight_variations.items()):
        agree_weighted = agree_normalised_weighted[['e1', 'e2', 'e3', 'e4']].dot(pd.Series(weights))
        disagree_weighted = disagree_normalised_weighted[['e1', 'e2', 'e3', 'e4']].dot(pd.Series(weights))
        net_scores = agree_weighted - disagree_weighted
        sorted_scores = sorted(net_scores, reverse=True)
        ax.plot(range(len(sorted_scores)), sorted_scores, marker='o', color=colors[i], label=labels[i])
    
    ax.set_title('Combined Sensitivity Analysis')
    ax.set_xlabel('Drug-Expert Combinations')
    ax.set_ylabel('Net Score')
    ax.grid(True)
    ax.legend()

# Create a new figure for the combined plot
fig, ax = plt.subplots(figsize=(12, 9))
plot_combined_sensitivity(agree_normalised_weighted, disagree_normalised_weighted, weight_variations, ax)
plt.tight_layout()
plt.savefig(f'{output_folder}/sensitivity_analysis_combined_all_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()