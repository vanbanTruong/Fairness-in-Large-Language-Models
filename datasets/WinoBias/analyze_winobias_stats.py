# Analyze the statistical results of WinoBias's Differential Metric Bias and build visualization charts
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the summary statistics data"""
    df = pd.read_excel(file_path, index_col=0)
    return df

def calculate_differential_bias(df, metric):
    """Calculate Differential Metric Bias (Δf) for a specific metric"""
    anti_scores = df.loc[['Anti_Type1', 'Anti_Type2'], metric].values
    pro_scores = df.loc[['Pro_Type1', 'Pro_Type2'], metric].values
    
    delta_f = np.mean(anti_scores) - np.mean(pro_scores)
    return delta_f

def calculate_statistical_tests(df, metric):
    """Calculate t-test and Mann-Whitney U test p-values for a specific metric"""
    anti_scores = df.loc[['Anti_Type1', 'Anti_Type2'], metric].values
    pro_scores = df.loc[['Pro_Type1', 'Pro_Type2'], metric].values
    
    # t-test
    t_stat, t_pval = stats.ttest_ind(anti_scores, pro_scores)
    
    # Mann-Whitney U test
    u_stat, u_pval = stats.mannwhitneyu(anti_scores, pro_scores, alternative='two-sided')
    
    return {
        't_test_p_value': t_pval,
        'mann_whitney_p_value': u_pval
    }

def calculate_cohens_d(df, metric):
    """Calculate Cohen's d effect size for a specific metric"""
    anti_scores = df.loc[['Anti_Type1', 'Anti_Type2'], metric].values
    pro_scores = df.loc[['Pro_Type1', 'Pro_Type2'], metric].values
    
    # Calculate means
    anti_mean = np.mean(anti_scores)
    pro_mean = np.mean(pro_scores)
    
    # Calculate pooled standard deviation
    anti_std = np.std(anti_scores, ddof=1)
    pro_std = np.std(pro_scores, ddof=1)
    n1, n2 = len(anti_scores), len(pro_scores)
    pooled_std = np.sqrt(((n1 - 1) * anti_std**2 + (n2 - 1) * pro_std**2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    cohens_d = (anti_mean - pro_mean) / pooled_std
    
    return cohens_d

def plot_score_distributions(df, metric, title):
    """Plot score distributions for anti-stereotypical and pro-stereotypical cases"""
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    anti_scores = df.loc[['Anti_Type1', 'Anti_Type2'], metric].values
    pro_scores = df.loc[['Pro_Type1', 'Pro_Type2'], metric].values
    
    # Create violin plot
    data = pd.DataFrame({
        'Score': np.concatenate([anti_scores, pro_scores]),
        'Type': ['Anti-stereotypical'] * len(anti_scores) + ['Pro-stereotypical'] * len(pro_scores)
    })
    
    sns.violinplot(data=data, x='Type', y='Score')
    plt.title(f'Distribution of {title} Scores')
    plt.ylabel(f'{title} Score')
    plt.grid(True, alpha=0.3)
    
    # Add mean markers
    plt.plot([0], np.mean(anti_scores), 'ro', label='Anti-stereotypical Mean')
    plt.plot([1], np.mean(pro_scores), 'bo', label='Pro-stereotypical Mean')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'wino_bias_{metric}_distributions.png')
    plt.close()

def analyze_metric(df, metric, title):
    """Analyze a specific metric and return results"""
    delta_f = calculate_differential_bias(df, metric)
    test_results = calculate_statistical_tests(df, metric)
    cohens_d = calculate_cohens_d(df, metric)
    
    # Plot distributions
    plot_score_distributions(df, metric, title)
    
    return {
        'metric': title,
        'differential_bias': delta_f,
        't_test_p_value': test_results['t_test_p_value'],
        'mann_whitney_p_value': test_results['mann_whitney_p_value'],
        'cohens_d': cohens_d
    }

def main():
    # Load data
    file_path = r"D:\Demo\dataset\data\WinoBias\analysis_results\summary_statistics.xlsx"
    df = load_data(file_path)
    
    # Define metrics to analyze
    metrics = {
        'avg_sentiment': 'Sentiment',
        'avg_toxicity': 'Toxicity',
        'avg_regard': 'Regard',
        'avg_gender_unigram': 'Gender Unigram',
        'avg_gender_max': 'Gender Max',
        'avg_gender_wavg': 'Gender Wavg'
    }
    
    # Analyze each metric
    all_results = []
    for metric, title in metrics.items():
        print(f"\nAnalyzing {title}...")
        results = analyze_metric(df, metric, title)
        all_results.append(results)
        
        # Print results for this metric
        print(f"\n{title} Analysis Results:")
        print("=" * 50)
        print(f"Differential Metric Bias (Δf): {results['differential_bias']:.4f}")
        print(f"t-test p-value: {results['t_test_p_value']:.4f}")
        print(f"Mann-Whitney U test p-value: {results['mann_whitney_p_value']:.4f}")
        print(f"Cohen's d: {results['cohens_d']:.4f}")
        print(f"Distribution plot saved as 'wino_bias_{metric}_distributions.png'")
    
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('wino_bias_statistical_results.csv', index=False)
    print("\nDetailed results saved to 'wino_bias_statistical_results.csv'")

if __name__ == "__main__":
    main() 