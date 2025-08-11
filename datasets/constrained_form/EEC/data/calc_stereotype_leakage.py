import pandas as pd
import numpy as np
from collections import Counter
from itertools import product

# Load EEC dataset
df = pd.read_csv('data/EEC/data/Equity-Evaluation-Corpus.csv')

# Define group words (WG) and trait words (WT)
group_words = df['Person'].unique().tolist()
trait_words = df['Emotion word'].dropna().unique().tolist()

# Filter to rows with non-empty emotion word (trait)
df_trait = df[df['Emotion word'].notnull() & (df['Emotion word'] != '')]

# Count co-occurrences (joint counts)
joint_counts = Counter(zip(df_trait['Person'], df_trait['Emotion word']))

total_joint = sum(joint_counts.values())

# Marginal counts
group_counts = Counter(df_trait['Person'])
trait_counts = Counter(df_trait['Emotion word'])
total_group = sum(group_counts.values())
total_trait = sum(trait_counts.values())

# Compute probabilities
P_joint = {(g, t): c / total_joint for (g, t), c in joint_counts.items()}
P_group = {g: group_counts[g] / total_group for g in group_words}
P_trait = {t: trait_counts[t] / total_trait for t in trait_words}

# Compute PMI for all observed pairs
PMI = {}
for (g, t), p_joint in P_joint.items():
    p_g = P_group[g]
    p_t = P_trait[t]
    if p_joint > 0 and p_g > 0 and p_t > 0:
        PMI[(g, t)] = np.log2(p_joint / (p_g * p_t))

# Print top 10 PMI pairs (should be close to zero if balanced)
print('Top 10 PMI(group, trait) pairs:')
for (g, t), pmi in sorted(PMI.items(), key=lambda x: -abs(x[1]))[:10]:
    print(f'{g} + {t}: {pmi:.4f}')

# Compute MI (Mutual Information)
MI = 0.0
for (g, t), p_joint in P_joint.items():
    p_g = P_group[g]
    p_t = P_trait[t]
    if p_joint > 0 and p_g > 0 and p_t > 0:
        MI += p_joint * np.log2(p_joint / (p_g * p_t))

print(f'\nCorpus-level Mutual Information (MI) between group and trait words: {MI:.6f} bits') 