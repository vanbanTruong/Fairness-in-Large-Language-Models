#WinBias————Representativeness Bias
import numpy as np
from collections import Counter
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def read_occupations(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def calculate_kl_divergence(p, q):
    """
    Calculate KL divergence between two probability distributions
    p: true distribution
    q: approximate distribution
    """
    epsilon = 1e-10  # Small value to avoid log(0)
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    return np.sum(p * np.log(p / q))

def create_visualizations(winobias_dist, bls_dist, kl_divergence):
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Set font sizes - titles to 20px, everything else to 18px
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Bar chart comparing distributions
    plt.subplot(2, 2, 1)
    df = pd.DataFrame({
        'Occupation': list(winobias_dist.keys()),
        'WinoBias': list(winobias_dist.values()),
        'BLS': list(bls_dist.values())
    })
    df_melted = pd.melt(df, id_vars=['Occupation'], 
                        value_vars=['WinoBias', 'BLS'],
                        var_name='Dataset', value_name='Probability')
    
    sns.barplot(data=df_melted, x='Occupation', y='Probability', hue='Dataset')
    plt.xticks(rotation=90, fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Occupation Distribution Comparison', fontsize=20)
    plt.xlabel('Occupation', fontsize=18)
    plt.ylabel('Probability', fontsize=18)
    plt.legend(fontsize=18)
    plt.tight_layout()
    
    # 2. Heatmap of differences
    plt.subplot(2, 2, 2)
    differences = {occ: abs(winobias_dist[occ] - bls_dist[occ]) for occ in winobias_dist.keys()}
    diff_df = pd.DataFrame(list(differences.items()), columns=['Occupation', 'Difference'])
    diff_df = diff_df.sort_values('Difference', ascending=False)
    
    sns.barplot(data=diff_df, x='Occupation', y='Difference')
    plt.xticks(rotation=90, fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Absolute Differences Between Distributions', fontsize=20)
    plt.xlabel('Occupation', fontsize=18)
    plt.ylabel('Absolute Difference', fontsize=18)
    plt.tight_layout()
    
    # 3. Pie charts for top occupations
    plt.subplot(2, 2, 3)
    top_n = 10
    top_winobias = dict(sorted(winobias_dist.items(), key=lambda x: x[1], reverse=True)[:top_n])
    plt.pie(top_winobias.values(), labels=top_winobias.keys(), autopct='%1.1f%%', textprops={'fontsize': 18})
    plt.title(f'Top {top_n} Occupations in WinoBias', fontsize=20)
    
    plt.subplot(2, 2, 4)
    top_bls = dict(sorted(bls_dist.items(), key=lambda x: x[1], reverse=True)[:top_n])
    plt.pie(top_bls.values(), labels=top_bls.keys(), autopct='%1.1f%%', textprops={'fontsize': 18})
    plt.title(f'Top {top_n} Occupations in BLS', fontsize=20)
    
    # Add KL divergence as text with larger font
    plt.figtext(0.5, 0.01, f'KL Divergence (Representativeness Bias): {kl_divergence:.4f}', 
                ha='center', fontsize=20, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('WinoBias/visualization/winobias_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Read occupations from WinoBias
    male_occupations = read_occupations('WinoBias/data/male_occupations.txt')
    female_occupations = read_occupations('WinoBias/data/female_occupations.txt')
    
    # Calculate WinoBias distribution (PD)
    all_occupations = male_occupations + female_occupations
    occupation_counts = Counter(all_occupations)
    total_occupations = len(all_occupations)
    winobias_dist = {occ: count/total_occupations for occ, count in occupation_counts.items()}
    
    # BLS data (PBLS) - This is a simplified example using 2019 BLS data
    # In reality, you would want to use the actual BLS data for these occupations
    bls_data = {
        'driver': 0.02,
        'supervisor': 0.015,
        'janitor': 0.01,
        'cook': 0.015,
        'mover': 0.005,
        'laborer': 0.01,
        'construction worker': 0.02,
        'chief': 0.005,
        'developer': 0.02,
        'carpenter': 0.01,
        'manager': 0.03,
        'lawyer': 0.01,
        'farmer': 0.005,
        'salesperson': 0.03,
        'physician': 0.01,
        'guard': 0.01,
        'analyst': 0.015,
        'mechanic': 0.015,
        'sheriff': 0.005,
        'CEO': 0.005,
        'attendant': 0.02,
        'cashier': 0.03,
        'teacher': 0.04,
        'nurse': 0.03,
        'assistant': 0.02,
        'secretary': 0.015,
        'auditor': 0.01,
        'cleaner': 0.015,
        'receptionist': 0.015,
        'clerk': 0.02,
        'counselor': 0.015,
        'designer': 0.015,
        'hairdresser': 0.01,
        'writer': 0.01,
        'housekeeper': 0.01,
        'baker': 0.005,
        'accountant': 0.015,
        'editor': 0.01,
        'librarian': 0.005,
        'tailor': 0.005
    }
    
    # Normalize BLS data to make it a proper probability distribution
    total_bls = sum(bls_data.values())
    bls_dist = {occ: freq/total_bls for occ, freq in bls_data.items()}
    
    # Calculate KL divergence
    # Convert distributions to lists in the same order
    occupations = sorted(set(list(winobias_dist.keys()) + list(bls_dist.keys())))
    winobias_values = [winobias_dist.get(occ, 0) for occ in occupations]
    bls_values = [bls_dist.get(occ, 0) for occ in occupations]
    
    kl_divergence = calculate_kl_divergence(winobias_values, bls_values)
    
    print(f"KL Divergence (Representativeness Bias): {kl_divergence:.4f}")
    
    # Create visualizations
    create_visualizations(winobias_dist, bls_dist, kl_divergence)
    
    # Print some statistics
    print("\nOccupation Distribution in WinoBias:")
    for occ, prob in sorted(winobias_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"{occ}: {prob:.4f}")
    
    print("\nReal-world Distribution (BLS):")
    for occ, prob in sorted(bls_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"{occ}: {prob:.4f}")

if __name__ == "__main__":
    main() 