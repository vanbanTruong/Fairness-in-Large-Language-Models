import pandas as pd
import numpy as np
from scipy.stats import entropy

# BLS 2024 population statistics
BLS_GENDER_DIST = {'male': 0.512, 'female': 0.488}  # binary only
dataset_gender_keys = ['male', 'female']

# BLS 2024 race distribution (only those present in EEC)
# 'black': 0.136 (African-American), 'white': 0.589 (European)
BLS_RACE_DIST = {'African-American': 0.136, 'European': 0.589}
dataset_race_keys = ['African-American', 'European']

# Load EEC dataset
df = pd.read_csv('data\EEC\data\Equity-Evaluation-Corpus.csv')

# Gender distribution in dataset (excluding NA)
gender_counts = df[df['Gender'].isin(dataset_gender_keys)]['Gender'].value_counts().reindex(dataset_gender_keys, fill_value=0)
gender_dist = gender_counts / gender_counts.sum()

# Race distribution in dataset (excluding NA)
race_counts = df[df['Race'].isin(dataset_race_keys)]['Race'].value_counts().reindex(dataset_race_keys, fill_value=0)
race_dist = race_counts / race_counts.sum()

# Prepare population distributions as arrays
bls_gender_dist = np.array([BLS_GENDER_DIST[k] for k in dataset_gender_keys])
dataset_gender_dist = gender_dist.values
bls_race_dist = np.array([BLS_RACE_DIST[k] for k in dataset_race_keys])
dataset_race_dist = race_dist.values

# KL divergence (add small epsilon to avoid log(0))
eps = 1e-12
gender_kl = entropy(dataset_gender_dist + eps, bls_gender_dist + eps)
race_kl = entropy(dataset_race_dist + eps, bls_race_dist + eps)

print('EEC Gender Distribution:', dict(zip(dataset_gender_keys, dataset_gender_dist.round(3))))
print('BLS 2024 Gender Distribution:', BLS_GENDER_DIST)
print('Representativeness Bias (KL Divergence, Gender):', round(gender_kl, 4))
print()
print('EEC Race Distribution:', dict(zip(dataset_race_keys, dataset_race_dist.round(3))))
print('BLS 2024 Race Distribution:', BLS_RACE_DIST)
print('Representativeness Bias (KL Divergence, Race):', round(race_kl, 4)) 