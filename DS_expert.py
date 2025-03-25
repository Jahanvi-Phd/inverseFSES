import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(42)

# Current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = "test_output"
os.makedirs(output_folder, exist_ok=True)

# Load Data
data = pd.read_csv("drug_data.csv")

# Define constants and functions
U = list(data['Drug'].unique())
E = ['Age', 'BP', 'Cholesterol', 'Na_to_K']
M = ['p', 'q', 'r']
weights = {'e1': 0.15, 'e2': 0.35, 'e3': 0.35, 'e4': 0.15}
expert_credibility = {'p': 0.9, 'q': 0.8, 'r': 0.7}

# Fuzzification functions
def fuzzify_age(age): return round(np.clip(age / 100, 0.2, 1.0), 2)
def fuzzify_na_to_k(value): return round(np.clip(value / 20, 0.2, 1.0), 2)
def fuzzify_bp(bp): return {'LOW': 0.3, 'NORMAL': 0.8, 'HIGH': 0.4}.get(bp, 0.5)
def fuzzify_cholesterol(ch): return {'NORMAL': 0.8, 'HIGH': 0.2}.get(ch, 0.5)

# Helper functions
def to_dataframe(ifses):
    return pd.DataFrame({
        'Drug-Expert': [f"{d},{e}" for d, _, e, _, _ in ifses],
        'e1': [p[0] for _, p, _, _, _ in ifses],
        'e2': [p[1] for _, p, _, _, _ in ifses],
        'e3': [p[2] for _, p, _, _, _ in ifses],
        'e4': [p[3] for _, p, _, _, _ in ifses],
        'Credibility': [c for _, _, _, c, _ in ifses]
    })

def row_normalize(df):
    norm_df = df.copy()
    for index, row in df.iterrows():
        row_sum_squares = np.sqrt((row[['e1', 'e2', 'e3', 'e4']] ** 2).sum())
        norm_df.loc[index, ['e1', 'e2', 'e3', 'e4']] = row[['e1', 'e2', 'e3', 'e4']] / row_sum_squares
    return norm_df

def weighted_sum(df):
    param_weighted = df[['e1', 'e2', 'e3', 'e4']].dot(pd.Series(weights))
    return param_weighted * df['Credibility']

# Main analysis function
def run_model(data, expert_credibility_dict):
    IFSES = []
    opinions = [0] * 100 + [1] * 100
    np.random.shuffle(opinions)
    
    for i, row in data.iterrows():
        drug = row['Drug']
        params = (fuzzify_age(row['Age']), 
                 fuzzify_bp(row['BP']), 
                 fuzzify_cholesterol(row['Cholesterol']), 
                 fuzzify_na_to_k(row['Na_to_K']))
        expert = np.random.choice(M)
        opinion = opinions[i]
        credibility = expert_credibility_dict[expert]
        IFSES.append((drug, params, expert, credibility, opinion))
    
    # Process data
    agree_IFSES = [entry for entry in IFSES if entry[4] == 1]
    disagree_IFSES = [entry for entry in IFSES if entry[4] == 0]
    
    agree_df = to_dataframe(agree_IFSES)
    disagree_df = to_dataframe(disagree_IFSES)
    
    agree_normalised = row_normalize(agree_df)
    disagree_normalised = row_normalize(disagree_df)
    
    agree_normalised['Weighted_Sum'] = weighted_sum(agree_normalised)
    disagree_normalised['Weighted_Sum'] = weighted_sum(disagree_normalised)
    
    net_scores = pd.DataFrame({
        'Drug-Expert': agree_normalised['Drug-Expert'],
        'Agree_Sum': agree_normalised['Weighted_Sum'],
        'Disagree_Sum': disagree_normalised['Weighted_Sum'],
    })
    net_scores['Net_Score'] = net_scores['Agree_Sum'] - net_scores['Disagree_Sum']
    
    return net_scores.sort_values(by='Net_Score', ascending=False)

# Run analyses
baseline = run_model(data, {'p': 1.0, 'q': 1.0, 'r': 1.0})
full_model = run_model(data, expert_credibility)
expert_p = run_model(data, {'p': 0.9, 'q': 1.0, 'r': 1.0})
expert_q = run_model(data, {'p': 1.0, 'q': 0.8, 'r': 1.0})
expert_r = run_model(data, {'p': 1.0, 'q': 1.0, 'r': 0.7})

# Create combined plots
plt.figure(figsize=(14, 10))

# First plot: Baseline vs Full Model
plt.subplot(2, 1, 1)
plt.plot(baseline['Net_Score'].values, 'b-o', label='Baseline (No Credibility)')
plt.plot(full_model['Net_Score'].values, 'g--s', label='With Expert Credibility')
plt.title('Model Comparison: Baseline vs Credibility-Weighted', fontsize=12)
plt.ylabel('Net Score')
plt.legend()
plt.grid(True)

# Second plot: Expert Sensitivity
plt.subplot(2, 1, 2)
plt.plot(expert_p['Net_Score'].values, 'r-o', label='Expert P (0.9)')
plt.plot(expert_q['Net_Score'].values, 'm--d', label='Expert Q (0.8)')
plt.plot(expert_r['Net_Score'].values, 'c-.X', label='Expert R (0.7)')
plt.title('Individual Expert Credibility Impact', fontsize=12)
plt.xlabel('Drug-Expert Combinations (Sorted by Net Score)')
plt.ylabel('Net Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'{output_folder}/combined_analysis_{timestamp}.png', dpi=300)
plt.show()

print("Analysis complete. Results saved to:", output_folder)