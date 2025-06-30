import compat_fix
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim.downloader as api
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
import scipy
import scipy.linalg

if not hasattr(scipy.linalg, 'triu'):
    def triu_replacement(m, k=0):
        """替代scipy.linalg.triu的实现，使用numpy"""
        return np.triu(m, k=k)
    
    # 使用monkey patching添加到scipy.linalg模块
    scipy.linalg.triu = triu_replacement
    print("Added triu replacement function to scipy.linalg")

# 添加模型管理功能
def get_model_path(model_name):
    """获取模型的本地保存路径"""
    # 创建models目录（如果不存在）
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return os.path.join(models_dir, f"{model_name}.model")

def load_or_download_model(model_name):
    """加载本地模型，如果不存在则下载并保存"""
    model_path = get_model_path(model_name)
    
    # 检查本地模型是否存在
    if os.path.exists(model_path):
        print(f"正在从本地加载模型: {model_path}")
        from gensim.models import KeyedVectors
        return KeyedVectors.load(model_path)
    
    # 如果本地不存在，则下载模型
    print(f"从服务器下载模型: {model_name}，这可能需要一些时间...")
    model = api.load(model_name)
    
    # 保存到本地
    print(f"将模型保存到本地: {model_path}")
    model.save(model_path)
    
    return model

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class GenderPolarityAnalyzer:
    """
    A class to calculate gender polarity of text using three different methods:
    1. Unigram Matching: Count male vs female tokens
    2. Gender-Max: Use the most gender-polarized word
    3. Gender-Wavg: Weighted average of gender polarities
    """
    
    def __init__(self, model_name='word2vec-google-news-300', use_debiased=True):
        """
        Initialize the gender polarity analyzer.
        
        Args:
            model_name: The name of the word embedding model to use
            use_debiased: Whether to try loading a debiased embedding
        """
        print("Initializing Gender Polarity Analyzer...")
        
        # Define gender-specific word lists
        self.male_tokens = set(['he', 'him', 'his', 'himself', 'man', 'men', "he's", 'boy', 'boys'])
        self.female_tokens = set(['she', 'her', 'hers', 'herself', 'woman', 'women', "she's", 'girl', 'girls'])
        
        # Load word embeddings
        print(f"Loading word embeddings: {model_name}")
        try:
            # First try to load debiased embeddings if requested
            if use_debiased:
                try:
                    # Check if we have debiased embeddings locally
                    debiased_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                              'models', 'debiased_word2vec.model')
                    if os.path.exists(debiased_path):
                        from gensim.models import KeyedVectors
                        self.model = KeyedVectors.load(debiased_path)
                        print("Loaded debiased Word2Vec from local file")
                    else:
                        # Try to download from GitHub
                        print("Debiased embeddings not found locally. Using standard embeddings.")
                        raise FileNotFoundError
                except Exception as e:
                    print(f"Could not load debiased embeddings: {e}")
                    # 使用修改后的加载函数
                    self.model = load_or_download_model(model_name)
            else:
                # 使用修改后的加载函数
                self.model = load_or_download_model(model_name)
                
            print(f"Word embeddings loaded with {len(self.model.key_to_index)} words")
        except Exception as e:
            print(f"Error loading word embeddings: {e}")
            print("Using a simplified version with random embeddings for demonstration")
            self.model = None
        
        # Calculate gender direction vector
        self.gender_direction = self._calculate_gender_direction()
        
    def _calculate_gender_direction(self):
        """Calculate the gender direction vector from female to male word embeddings"""
        if self.model is None:
            # Return a random vector if no embeddings are available
            return np.random.randn(300)
        
        # Define pairs of gender-specific words
        gender_pairs = [
            ('woman', 'man'),
            ('girl', 'boy'),
            ('she', 'he'),
            ('her', 'him'),
            ('mother', 'father'),
            ('daughter', 'son'),
            ('wife', 'husband'),
            ('sister', 'brother')
        ]
        
        # Calculate gender direction as average of differences between female and male words
        directions = []
        for female, male in gender_pairs:
            if female in self.model and male in self.model:
                # Female - male to get the gender direction (positive is female, negative is male)
                directions.append(self.model[female] - self.model[male])
        
        if not directions:
            print("Warning: Could not calculate gender direction from embeddings")
            return np.random.randn(self.model.vector_size)
        
        # Average all directions and normalize
        gender_direction = np.mean(directions, axis=0)
        return gender_direction / np.linalg.norm(gender_direction)
    
    def preprocess_text(self, text):
        """Clean and tokenize the text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords (optional, commented out as gender words are often stopwords)
        # stop_words = set(stopwords.words('english'))
        # tokens = [word for word in tokens if word not in stop_words]
        
        return tokens
    
    def calculate_word_gender_polarity(self, word):
        """
        Calculate gender polarity of a single word using projection onto gender direction
        
        Returns:
            float: A score between -1 (male) and 1 (female)
        """
        if self.model is None or word not in self.model:
            # If word not in vocabulary, return 0 (neutral)
            return 0
        
        # Get word vector
        word_vec = self.model[word]
        word_vec_norm = word_vec / np.linalg.norm(word_vec)
        
        # Project onto gender direction (dot product)
        projection = np.dot(word_vec_norm, self.gender_direction)
        
        return projection
    
    def unigram_matching(self, text):
        """
        Calculate gender polarity using unigram matching of gendered words
        
        Args:
            text: Input text string
            
        Returns:
            score: A value from -1 (male) to 1 (female)
            label: 'male', 'female', or 'neutral'
        """
        tokens = self.preprocess_text(text)
        
        # Count male and female tokens
        male_count = sum(1 for token in tokens if token in self.male_tokens)
        female_count = sum(1 for token in tokens if token in self.female_tokens)
        
        # Determine polarity
        if male_count > female_count:
            score = -1 * (male_count - female_count) / (male_count + female_count + 1)
            label = 'male'
        elif female_count > male_count:
            score = (female_count - male_count) / (male_count + female_count + 1)
            label = 'female'
        else:
            score = 0
            label = 'neutral'
            
        return score, label
    
    def gender_max(self, text):
        """
        Calculate gender polarity using the most gender-polarized word
        
        Args:
            text: Input text string
            
        Returns:
            score: A value from -1 (male) to 1 (female)
            label: 'male', 'female', or 'neutral'
        """
        tokens = self.preprocess_text(text)
        
        # Calculate gender polarity for each word
        polarities = [(token, self.calculate_word_gender_polarity(token)) for token in tokens]
        
        # Find the most polarized word (highest absolute value)
        if not polarities:
            return 0, 'neutral'
        
        max_word, max_polarity = max(polarities, key=lambda x: abs(x[1]))
        
        # Determine label based on thresholds from the paper
        if max_polarity <= -0.25:
            label = 'male'
        elif max_polarity >= 0.25:
            label = 'female'
        else:
            label = 'neutral'
            
        return max_polarity, label
    
    def gender_wavg(self, text):
        """
        Calculate gender polarity using weighted average of word polarities
        
        Args:
            text: Input text string
            
        Returns:
            score: A value from -1 (male) to 1 (female)
            label: 'male', 'female', or 'neutral'
        """
        tokens = self.preprocess_text(text)
        
        # Calculate gender polarity for each word
        polarities = [self.calculate_word_gender_polarity(token) for token in tokens]
        
        # Skip if no valid polarities
        if not polarities or all(p == 0 for p in polarities):
            return 0, 'neutral'
        
        # Calculate weighted average according to formula in the paper
        # Gender-Wavg = Σ(sgn(cⱼ) * cⱼ²) / Σ(|cⱼ|)
        numerator = sum(np.sign(p) * p * p for p in polarities)
        denominator = sum(abs(p) for p in polarities)
        
        if denominator == 0:
            wavg_polarity = 0
        else:
            wavg_polarity = numerator / denominator
        
        # Determine label based on thresholds from the paper
        if wavg_polarity <= -0.25:
            label = 'male'
        elif wavg_polarity >= 0.25:
            label = 'female'
        else:
            label = 'neutral'
            
        return wavg_polarity, label
    
    def analyze_text(self, text):
        """
        Analyze text using all three gender polarity methods
        
        Args:
            text: Input text string
            
        Returns:
            dict: Results from all three methods
        """
        results = {
            'text': text,
            'unigram': {},
            'gender_max': {},
            'gender_wavg': {}
        }
        
        # Calculate using all three methods
        unigram_score, unigram_label = self.unigram_matching(text)
        max_score, max_label = self.gender_max(text)
        wavg_score, wavg_label = self.gender_wavg(text)
        
        # Store results
        results['unigram'] = {'score': unigram_score, 'label': unigram_label}
        results['gender_max'] = {'score': max_score, 'label': max_label}
        results['gender_wavg'] = {'score': wavg_score, 'label': wavg_label}
        
        return results

def analyze_json_file(file_path, analyzer):
    """
    Analyze all text samples in a JSON file for gender polarity
    
    Args:
        file_path: Path to the JSON file
        analyzer: GenderPolarityAnalyzer instance
        
    Returns:
        dict: Results for each category
    """
    # Load data
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare results storage
    results = {}
    
    # Process each category
    for category, entries in tqdm(data.items(), desc="Analyzing categories"):
        category_results = {
            'unigram': {'male': 0, 'neutral': 0, 'female': 0, 'scores': []},
            'gender_max': {'male': 0, 'neutral': 0, 'female': 0, 'scores': []},
            'gender_wavg': {'male': 0, 'neutral': 0, 'female': 0, 'scores': []}
        }
        
        # Process each text
        text_count = 0
        for person, texts in entries.items():
            for text in texts:
                # Analyze the text
                analysis = analyzer.analyze_text(text)
                
                # Update counts
                category_results['unigram'][analysis['unigram']['label']] += 1
                category_results['gender_max'][analysis['gender_max']['label']] += 1
                category_results['gender_wavg'][analysis['gender_wavg']['label']] += 1
                
                # Store scores
                category_results['unigram']['scores'].append(analysis['unigram']['score'])
                category_results['gender_max']['scores'].append(analysis['gender_max']['score'])
                category_results['gender_wavg']['scores'].append(analysis['gender_wavg']['score'])
                
                text_count += 1
        
        # Calculate averages
        for method in ['unigram', 'gender_max', 'gender_wavg']:
            category_results[method]['avg_score'] = sum(category_results[method]['scores']) / text_count if text_count > 0 else 0
        
        results[category] = category_results
    
    return results

def visualize_results(results, file_name):
    """Create visualizations to compare gender polarity across categories"""
    print("Generating visualization charts...")
    categories = list(results.keys())
    methods = ['unigram', 'gender_max', 'gender_wavg']
    method_names = ['Unigram Matching', 'Gender-Max', 'Gender-Wavg']
    
    # Create a 3x1 plot grid
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # Muted color scheme
    colors = {
        'male': '#6C8EAD',    # Blue
        'neutral': '#D8D8D8', # Light gray
        'female': '#B6A39E'   # Pink/beige
    }
    
    # Plot distribution for each method
    for i, (method, method_name) in enumerate(zip(methods, method_names)):
        ax = axes[i]
        
        x = np.arange(len(categories))
        width = 0.25
        
        # Calculate percentages
        male_pcts = [results[cat][method]['male'] / sum(results[cat][method][label] for label in ['male', 'neutral', 'female']) * 100 for cat in categories]
        neutral_pcts = [results[cat][method]['neutral'] / sum(results[cat][method][label] for label in ['male', 'neutral', 'female']) * 100 for cat in categories]
        female_pcts = [results[cat][method]['female'] / sum(results[cat][method][label] for label in ['male', 'neutral', 'female']) * 100 for cat in categories]
        
        # Create stacked bars
        ax.bar(x, male_pcts, width, label='Male', color=colors['male'])
        ax.bar(x, neutral_pcts, width, bottom=male_pcts, label='Neutral', color=colors['neutral'])
        ax.bar(x, female_pcts, width, bottom=[m+n for m,n in zip(male_pcts, neutral_pcts)], label='Female', color=colors['female'])
        
        # Add average score values as a line
        avg_scores = [results[cat][method]['avg_score'] for cat in categories]
        ax2 = ax.twinx()
        ax2.plot(x, avg_scores, 'o-', color='#9C6644', linewidth=2, label='Avg Score')
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel('Average Score (-1=male to 1=female)')
        
        # Add a horizontal line at y=0
        ax2.axhline(y=0, color='#D8D2CB', linestyle='-', alpha=0.5)
        
        # Formatting
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'Gender Polarity Distribution: {method_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#E6E2DD')
        
    # Style settings
    plt.rcParams['font.family'] = 'serif'
    fig.patch.set_facecolor('#F5F3F1')  # Light beige background
    for ax in axes:
        ax.set_facecolor('#F9F7F6')  # Very light gray background
        for spine in ax.spines.values():
            spine.set_color('#E6E2DD')  # Light gray border
    
    plt.tight_layout()
    output_file = f"{os.path.splitext(os.path.basename(file_name))[0]}_gender_polarity.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved as {output_file}")

def save_results_to_csv(results, file_name):
    """Save analysis results to CSV files"""
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    
    # Prepare data for CSV export
    rows = []
    for category, data in results.items():
        row = {'category': category}
        
        for method in ['unigram', 'gender_max', 'gender_wavg']:
            row[f'{method}_male'] = data[method]['male']
            row[f'{method}_neutral'] = data[method]['neutral']
            row[f'{method}_female'] = data[method]['female']
            row[f'{method}_avg_score'] = data[method]['avg_score']
        
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    output_file = f"{base_name}_gender_polarity_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def analyze_toxicity_prompts(file_path, analyzer):
    """
    Analyze gender polarity in RealToxicityPrompts dataset
    
    Args:
        file_path: Path to the TXT file
        analyzer: GenderPolarityAnalyzer instance
        
    Returns:
        dict: Results for the entire dataset
    """
    print(f"Loading RealToxicityPrompts data from {file_path}")
    
    # Read TXT file
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]  # 读取非空行
    
    # Initialize results storage
    results = {
        'unigram': {'male': 0, 'neutral': 0, 'female': 0, 'scores': []},
        'gender_max': {'male': 0, 'neutral': 0, 'female': 0, 'scores': []},
        'gender_wavg': {'male': 0, 'neutral': 0, 'female': 0, 'scores': []}
    }
    
    # Process each text
    total_texts = len(texts)
    print(f"Processing {total_texts} texts...")
    
    for text in tqdm(texts, desc="Analyzing texts"):
        # Analyze the text
        analysis = analyzer.analyze_text(text)
        
        # Update counts
        results['unigram'][analysis['unigram']['label']] += 1
        results['gender_max'][analysis['gender_max']['label']] += 1
        results['gender_wavg'][analysis['gender_wavg']['label']] += 1
        
        # Store scores
        results['unigram']['scores'].append(analysis['unigram']['score'])
        results['gender_max']['scores'].append(analysis['gender_max']['score'])
        results['gender_wavg']['scores'].append(analysis['gender_wavg']['score'])
    
    # Calculate averages
    for method in ['unigram', 'gender_max', 'gender_wavg']:
        results[method]['avg_score'] = sum(results[method]['scores']) / total_texts
    
    return results

def visualize_toxicity_results(results, file_name):
    """Create visualizations for RealToxicityPrompts dataset results"""
    print("Generating visualization charts...")
    methods = ['unigram', 'gender_max', 'gender_wavg']
    method_names = ['Unigram Matching', 'Gender-Max', 'Gender-Wavg']
    
    # Set global font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans', 'Bitstream Vera Sans']  # 指定多个无衬线字体作为备选
    plt.rcParams['mathtext.fontset'] = 'dejavusans'  # 数学字体也使用无衬线
    
    # Muted color scheme
    colors = {
        'male': '#6C8EAD',    # Blue
        'neutral': '#D8D8D8', # Light gray
        'female': '#B6A39E'   # Pink/beige
    }
    
    # Create separate plot for each method
    for method, method_name in zip(methods, method_names):
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate percentages
        total = sum(results[method][label] for label in ['male', 'neutral', 'female'])
        male_pct = results[method]['male'] / total * 100
        neutral_pct = results[method]['neutral'] / total * 100
        female_pct = results[method]['female'] / total * 100
        
        # Create stacked bars
        ax.bar(0, male_pct, width=0.5, label='Male', color=colors['male'])
        ax.bar(0, neutral_pct, width=0.5, bottom=male_pct, label='Neutral', color=colors['neutral'])
        ax.bar(0, female_pct, width=0.5, bottom=male_pct+neutral_pct, label='Female', color=colors['female'])
        
        # Add average score value as a line
        avg_score = results[method]['avg_score']
        ax2 = ax.twinx()
        ax2.plot(0, avg_score, 'o', color='#9C6644', markersize=10, label='Avg Score')
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel('Average Score (-1=male to 1=female)')
        
        # Add a horizontal line at y=0
        ax2.axhline(y=0, color='#D8D2CB', linestyle='-', alpha=0.5)
        
        # Formatting
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'Gender Polarity Distribution: {method_name}')
        ax.set_xticks([])  # Remove x-axis ticks
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#E6E2DD')
        
        # Add percentage labels
        ax.text(0, male_pct/2, f'{male_pct:.1f}%', ha='center', va='center')
        ax.text(0, male_pct + neutral_pct/2, f'{neutral_pct:.1f}%', ha='center', va='center')
        ax.text(0, male_pct + neutral_pct + female_pct/2, f'{female_pct:.1f}%', ha='center', va='center')
        
        # Style settings
        fig.patch.set_facecolor('#F5F3F1')  # Light beige background
        ax.set_facecolor('#F9F7F6')  # Very light gray background
        for spine in ax.spines.values():
            spine.set_color('#E6E2DD')  # Light gray border
        
        plt.tight_layout()
        # Save each plot separately
        output_file = f"{os.path.splitext(os.path.basename(file_name))[0]}_{method}_gender_polarity.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved as {output_file}")

def save_toxicity_results_to_csv(results, file_name):
    """Save RealToxicityPrompts analysis results to CSV file"""
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    
    # Prepare data for CSV export
    rows = []
    for method in ['unigram', 'gender_max', 'gender_wavg']:
        row = {
            'method': method,
            'male_count': results[method]['male'],
            'neutral_count': results[method]['neutral'],
            'female_count': results[method]['female'],
            'male_percentage': results[method]['male'] / sum(results[method][label] for label in ['male', 'neutral', 'female']) * 100,
            'neutral_percentage': results[method]['neutral'] / sum(results[method][label] for label in ['male', 'neutral', 'female']) * 100,
            'female_percentage': results[method]['female'] / sum(results[method][label] for label in ['male', 'neutral', 'female']) * 100,
            'avg_score': results[method]['avg_score']
        }
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    output_file = f"{base_name}_gender_polarity_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def analyze_trustgpt_prompts(file_path, analyzer, max_samples=100):
    """
    Analyze gender polarity in TrustGPT dataset
    
    Args:
        file_path: Path to the JSON file
        analyzer: GenderPolarityAnalyzer instance
        max_samples: Maximum number of samples to analyze
        
    Returns:
        dict: Results for the entire dataset
    """
    print(f"Loading TrustGPT data from {file_path}")
    
    # Read JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize results storage
    results = {
        'unigram': {'male': 0, 'neutral': 0, 'female': 0, 'scores': []},
        'gender_max': {'male': 0, 'neutral': 0, 'female': 0, 'scores': []},
        'gender_wavg': {'male': 0, 'neutral': 0, 'female': 0, 'scores': []}
    }
    
    # Prepare detailed results for CSV
    detailed_results = []
    
    # Process texts based on file structure
    texts = []
    if isinstance(data, dict):  # For bias_prompts.json
        for category in data.values():
            texts.extend(category)
    else:  # For toxicity_prompts.json and value_alignment_prompts.json
        texts = data
    
    # Limit number of samples
    texts = texts[:max_samples]
    total_texts = len(texts)
    print(f"Processing {total_texts} texts...")
    
    # Process each text
    for i, text in enumerate(tqdm(texts, desc="Analyzing texts")):
        # Analyze the text
        analysis = analyzer.analyze_text(text)
        
        # Update counts
        results['unigram'][analysis['unigram']['label']] += 1
        results['gender_max'][analysis['gender_max']['label']] += 1
        results['gender_wavg'][analysis['gender_wavg']['label']] += 1
        
        # Store scores
        results['unigram']['scores'].append(analysis['unigram']['score'])
        results['gender_max']['scores'].append(analysis['gender_max']['score'])
        results['gender_wavg']['scores'].append(analysis['gender_wavg']['score'])
        
        # Store detailed results
        detailed_results.append({
            'text': text,
            'unigram_score': analysis['unigram']['score'],
            'unigram_label': analysis['unigram']['label'],
            'gender_max_score': analysis['gender_max']['score'],
            'gender_max_label': analysis['gender_max']['label'],
            'gender_wavg_score': analysis['gender_wavg']['score'],
            'gender_wavg_label': analysis['gender_wavg']['label']
        })
        
        # Save results every 10 sentences
        if (i + 1) % 10 == 0 or i == total_texts - 1:
            save_trustgpt_detailed_results(detailed_results, file_path, i + 1)
    
    # Calculate averages
    for method in ['unigram', 'gender_max', 'gender_wavg']:
        results[method]['avg_score'] = sum(results[method]['scores']) / total_texts
    
    return results

def save_trustgpt_detailed_results(results, file_path, sample_count):
    """Save detailed TrustGPT analysis results to CSV file by appending to a single file"""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f"{base_name}_detailed_results.csv"
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # If file exists, append without header, otherwise create new file with header
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)
    
    print(f"Saved results for {sample_count} samples to {output_file}")

def main(dataset='all', max_samples=100):
    """
    Main function to run the gender polarity analysis.
    
    Args:
        dataset: Which dataset to analyze. Options:
            - 'all': Analyze all datasets
            - 'bold': Analyze only BOLD dataset
            - 'toxicity': Analyze only RealToxicityPrompts dataset
            - 'trustgpt': Analyze only TrustGPT dataset
        max_samples: Maximum number of samples to analyze for TrustGPT dataset
    """
    # Path to the JSON files
    files = [
        "data/BOLD/data/wikipedia/gender_wiki.json",
        "data/BOLD/data/wikipedia/political_ideology_wiki.json",
        "data/BOLD/data/wikipedia/profession_wiki.json",
        "data/BOLD/data/wikipedia/race_wiki.json",
        "data/BOLD/data/wikipedia/religious_ideology_wiki.json"
    ]
    
    # Initialize analyzer
    analyzer = GenderPolarityAnalyzer()
    
    # Process BOLD dataset files
    if dataset in ['all', 'bold']:
        print("\n" + "="*80)
        print("Processing BOLD Dataset")
        print("="*80)
        for file_path in tqdm(files, desc="Processing BOLD files"):
            if os.path.exists(file_path):
                print(f"\nAnalyzing file: {file_path}...")
                results = analyze_json_file(file_path, analyzer)
                
                # Print results summary
                print(f"\nGender Polarity Results for {os.path.basename(file_path)}:")
                for category, data in results.items():
                    print(f"\n{category}:")
                    for method, method_data in data.items():
                        total = sum(method_data[label] for label in ['male', 'neutral', 'female'])
                        print(f"  {method.upper()}:")
                        print(f"    Male: {method_data['male']} ({method_data['male']/total*100:.2f}%)")
                        print(f"    Neutral: {method_data['neutral']} ({method_data['neutral']/total*100:.2f}%)")
                        print(f"    Female: {method_data['female']} ({method_data['female']/total*100:.2f}%)")
                        print(f"    Average Score: {method_data['avg_score']:.3f}")
                
                # Save results
                save_results_to_csv(results, file_path)
                
                # Create visualizations
                visualize_results(results, file_path)
            else:
                print(f"File not found: {file_path}")
    
    # Process RealToxicityPrompts dataset
    if dataset in ['all', 'toxicity']:
        print("\n" + "="*80)
        print("Processing RealToxicityPrompts Dataset")
        print("="*80)
        toxicity_file = "data/RealToxicityPrompts/merged.txt"
        if os.path.exists(toxicity_file):
            print(f"\nAnalyzing RealToxicityPrompts dataset: {toxicity_file}...")
            results = analyze_toxicity_prompts(toxicity_file, analyzer)
            
            # Print results summary
            print("\nGender Polarity Results for RealToxicityPrompts:")
            for method, method_data in results.items():
                total = sum(method_data[label] for label in ['male', 'neutral', 'female'])
                print(f"\n{method.upper()}:")
                print(f"  Male: {method_data['male']} ({method_data['male']/total*100:.2f}%)")
                print(f"  Neutral: {method_data['neutral']} ({method_data['neutral']/total*100:.2f}%)")
                print(f"  Female: {method_data['female']} ({method_data['female']/total*100:.2f}%)")
                print(f"  Average Score: {method_data['avg_score']:.3f}")
            
            # Save results
            save_toxicity_results_to_csv(results, toxicity_file)
            
            # Create visualizations
            visualize_toxicity_results(results, toxicity_file)
        else:
            print(f"RealToxicityPrompts file not found: {toxicity_file}")
    
    # Process TrustGPT dataset
    if dataset in ['all', 'trustgpt']:
        print("\n" + "="*80)
        print("Processing TrustGPT Dataset")
        print("="*80)
        trustgpt_files = [
            "data/TrustGPT/output/bias_prompts.json",
            "data/TrustGPT/output/toxicity_prompts.json",
            "data/TrustGPT/output/value_alignment_prompts.json"
        ]
        
        for file_path in trustgpt_files:
            if os.path.exists(file_path):
                print(f"\nAnalyzing TrustGPT file: {file_path}...")
                results = analyze_trustgpt_prompts(file_path, analyzer, max_samples)
                
                # Print results summary
                print(f"\nGender Polarity Results for {os.path.basename(file_path)}:")
                for method, method_data in results.items():
                    total = sum(method_data[label] for label in ['male', 'neutral', 'female'])
                    print(f"\n{method.upper()}:")
                    print(f"  Male: {method_data['male']} ({method_data['male']/total*100:.2f}%)")
                    print(f"  Neutral: {method_data['neutral']} ({method_data['neutral']/total*100:.2f}%)")
                    print(f"  Female: {method_data['female']} ({method_data['female']/total*100:.2f}%)")
                    print(f"  Average Score: {method_data['avg_score']:.3f}")
                
                # Save results
                save_toxicity_results_to_csv(results, file_path)
                
                # Create visualizations
                visualize_toxicity_results(results, file_path)
            else:
                print(f"TrustGPT file not found: {file_path}")

if __name__ == "__main__":
    # 'all', 'bold', 'toxicity', 'trustgpt'
    main(dataset='trustgpt', max_samples=100)  