# WinoBias_Differential Matric Bias
import os
from text_analysis_pipeline import TextAnalysisPipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def analyze_winobias_file(file_path, pipeline, output_dir):
    """
    Analyze a single WinoBias file and save results.
    
    Args:
        file_path: Path to the WinoBias file
        pipeline: TextAnalysisPipeline instance
        output_dir: Directory to save results
    """
    print(f"\nAnalyzing file: {file_path}")
    
    # Create output filename
    base_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_analysis.csv")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    
    # Analyze each text
    results = []
    for text in tqdm(texts, desc="Analyzing texts"):
        analysis = pipeline.analyze_text(text)
        # Flatten the nested dictionary structure
        flat_result = {
            'text': text,
            'sentiment_score': analysis['sentiment']['scores']['compound'],
            'sentiment_label': analysis['sentiment']['sentiment'],
            'toxicity_score': analysis['toxicity']['scores']['TOXICITY'],
            'severe_toxicity_score': analysis['toxicity']['scores']['SEVERE_TOXICITY'],
            'identity_attack_score': analysis['toxicity']['scores']['IDENTITY_ATTACK'],
            'insult_score': analysis['toxicity']['scores']['INSULT'],
            'threat_score': analysis['toxicity']['scores']['THREAT'],
            'regard_label': analysis['regard']['label'],
            'regard_score': analysis['regard']['score'],
            'gender_unigram_score': analysis['gender_polarity']['unigram']['score'],
            'gender_unigram_label': analysis['gender_polarity']['unigram']['label'],
            'gender_max_score': analysis['gender_polarity']['gender_max']['score'],
            'gender_max_label': analysis['gender_polarity']['gender_max']['label'],
            'gender_wavg_score': analysis['gender_polarity']['gender_wavg']['score'],
            'gender_wavg_label': analysis['gender_polarity']['gender_wavg']['label']
        }
        results.append(flat_result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return df

def analyze_winobias_dataset(data_dir, output_dir):
    """
    Analyze all WinoBias files in the specified directory.
    
    Args:
        data_dir: Directory containing WinoBias files
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = TextAnalysisPipeline()
    
    # Define file patterns
    file_patterns = [
        'anti_stereotyped_type1.txt.dev',
        'anti_stereotyped_type1.txt.test',
        'anti_stereotyped_type2.txt.dev',
        'anti_stereotyped_type2.txt.test',
        'pro_stereotyped_type1.txt.dev',
        'pro_stereotyped_type1.txt.test',
        'pro_stereotyped_type2.txt.dev',
        'pro_stereotyped_type2.txt.test'
    ]
    
    # Analyze each file
    all_results = {}
    for pattern in file_patterns:
        file_path = os.path.join(data_dir, pattern)
        if os.path.exists(file_path):
            results_df = analyze_winobias_file(file_path, pipeline, output_dir)
            all_results[pattern] = results_df
        else:
            print(f"Warning: File not found: {file_path}")
    
    # Create summary visualizations
    create_summary_visualizations(all_results, output_dir)
    
    return all_results

def create_summary_visualizations(results, output_dir):
    """
    Create summary visualizations of the analysis results.
    
    Args:
        results: Dictionary of DataFrames containing analysis results
        output_dir: Directory to save visualizations
    """
    # Set style
    plt.style.use('seaborn-v0_8')  # Updated to use the new style name
    
    # 1. Sentiment Analysis Comparison
    plt.figure(figsize=(12, 6))
    sentiment_data = []
    for file_name, df in results.items():
        sentiment_data.append({
            'file': file_name,
            'positive': (df['sentiment_label'] == 'Positive').mean(),
            'neutral': (df['sentiment_label'] == 'Neutral').mean(),
            'negative': (df['sentiment_label'] == 'Negative').mean()
        })
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df.set_index('file').plot(kind='bar', stacked=True)
    plt.title('Sentiment Distribution Across Files')
    plt.xlabel('File')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
    plt.close()
    
    # 2. Toxicity Analysis Comparison
    plt.figure(figsize=(12, 6))
    toxicity_data = []
    for file_name, df in results.items():
        toxicity_data.append({
            'file': file_name,
            'toxicity': df['toxicity_score'].mean(),
            'severe_toxicity': df['severe_toxicity_score'].mean(),
            'identity_attack': df['identity_attack_score'].mean(),
            'insult': df['insult_score'].mean(),
            'threat': df['threat_score'].mean()
        })
    toxicity_df = pd.DataFrame(toxicity_data)
    toxicity_df.set_index('file').plot(kind='bar')
    plt.title('Average Toxicity Scores Across Files')
    plt.xlabel('File')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'toxicity_scores.png'))
    plt.close()
    
    # 3. Gender Polarity Comparison
    plt.figure(figsize=(12, 6))
    gender_data = []
    for file_name, df in results.items():
        gender_data.append({
            'file': file_name,
            'unigram': df['gender_unigram_score'].mean(),
            'gender_max': df['gender_max_score'].mean(),
            'gender_wavg': df['gender_wavg_score'].mean()
        })
    gender_df = pd.DataFrame(gender_data)
    gender_df.set_index('file').plot(kind='bar')
    plt.title('Average Gender Polarity Scores Across Files')
    plt.xlabel('File')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_polarity_scores.png'))
    plt.close()
    
    # 4. Create summary statistics
    summary_stats = []
    for file_name, df in results.items():
        stats = {
            'file': file_name,
            'num_texts': len(df),
            'avg_sentiment': df['sentiment_score'].mean(),
            'avg_toxicity': df['toxicity_score'].mean(),
            'avg_regard': df['regard_score'].mean(),
            'avg_gender_unigram': df['gender_unigram_score'].mean(),
            'avg_gender_max': df['gender_max_score'].mean(),
            'avg_gender_wavg': df['gender_wavg_score'].mean()
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)

def main():
    # Define directories
    data_dir = "WinoBias/data"
    output_dir = "WinoBias/analysis_results"
    
    # Analyze dataset
    results = analyze_winobias_dataset(data_dir, output_dir)
    
    print("\nAnalysis complete! Results saved in:", output_dir)
    print("Generated files:")
    print("1. Individual analysis CSV files for each WinoBias file")
    print("2. sentiment_distribution.png")
    print("3. toxicity_scores.png")
    print("4. gender_polarity_scores.png")
    print("5. summary_statistics.csv")

if __name__ == "__main__":
    main() 