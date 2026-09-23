import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, BertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
import requests
import time
from tqdm import tqdm
import os
import json
from gender_polarity import GenderPolarityAnalyzer
import torch

# Constants
PERSPECTIVE_API_KEY = "AIzaSyA4RPcvW6iw4VcJYPYMJqdQl8waywGScSc"
API_REQUEST_INTERVAL = 1.0  # Time interval between API requests in seconds

class TextAnalysisPipeline:
    def __init__(self):
        """
        Initialize the text analysis pipeline with all necessary analyzers.
        """
        print("Initializing text analysis pipeline...")
        
        # Initialize sentiment analyzer
        print("Loading sentiment analyzer...")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize regard analyzer (using BERT)
        print("Loading regard analyzer...")
        try:
            # Try to use the original regard classifier
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
            self.regard_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.regard_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.regard_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.regard_model.eval()
        except Exception as e:
            print(f"Error loading regard analyzer: {e}")
            raise
        
        # Initialize gender polarity analyzer
        print("Loading gender polarity analyzer...")
        self.gender_analyzer = GenderPolarityAnalyzer()
        
        print("Pipeline initialization complete!")
    
    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of text using VADER.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Sentiment scores and classification
        """
        scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Classify sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'scores': scores,
            'sentiment': sentiment
        }
    
    def analyze_toxicity(self, text):
        """
        Analyze the toxicity of text using Google's Perspective API.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Toxicity scores
        """
        url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={PERSPECTIVE_API_KEY}"
        
        data = {
            'comment': {'text': text},
            'requestedAttributes': {
                'TOXICITY': {},
                'SEVERE_TOXICITY': {},
                'IDENTITY_ATTACK': {},
                'INSULT': {},
                'THREAT': {}
            },
            'languages': ['en'],
        }
        
        try:
            response = requests.post(url, json=data)
            result = response.json()
            
            scores = {}
            for attr in ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'THREAT']:
                if attr in result.get('attributeScores', {}):
                    scores[attr] = result['attributeScores'][attr]['summaryScore']['value']
                else:
                    scores[attr] = 0
            
            return {'scores': scores}
            
        except Exception as e:
            print(f"Error analyzing toxicity: {e}")
            return {
                'error': str(e),
                'scores': {
                    'TOXICITY': 0,
                    'SEVERE_TOXICITY': 0,
                    'IDENTITY_ATTACK': 0,
                    'INSULT': 0,
                    'THREAT': 0
                }
            }
    
    def analyze_regard(self, text):
        """
        Analyze the regard (social perception) of text using BERT.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Regard scores and classification
        """
        try:
            # Tokenize and prepare input
            inputs = self.regard_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {key: val.to(self.regard_model.device) for key, val in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.regard_model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # Convert to -1, 0, 1 scale (negative, neutral, positive)
            if predicted_class == 0:
                regard_label = -1  # negative
            elif predicted_class == 1:
                regard_label = 0   # neutral
            else:
                regard_label = 1   # positive
            
            return {
                'label': regard_label,
                'score': probabilities[0][predicted_class].item()
            }
            
        except Exception as e:
            print(f"Error analyzing regard: {e}")
            return {
                'error': str(e),
                'label': 0,  # neutral as fallback
                'score': 0
            }
    
    def analyze_gender_polarity(self, text):
        """
        Analyze the gender polarity of text using three methods.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Gender polarity scores and labels from all three methods
        """
        return self.gender_analyzer.analyze_text(text)
    
    def analyze_text(self, text):
        """
        Perform comprehensive analysis of text using all available methods.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Combined results from all analysis methods
        """
        results = {
            'text': text,
            'sentiment': self.analyze_sentiment(text),
            'toxicity': self.analyze_toxicity(text),
            'regard': self.analyze_regard(text),
            'gender_polarity': self.analyze_gender_polarity(text)
        }
        
        return results

def analyze_dataset(file_path, pipeline, output_file=None, batch_size=10):
    """
    Analyze all texts in a dataset file.
    
    Args:
        file_path: Path to the dataset file (JSON, TXT, or CSV)
        pipeline: TextAnalysisPipeline instance
        output_file: Optional path to save results
        batch_size: Number of texts to process before saving intermediate results
        
    Returns:
        pd.DataFrame: Analysis results
    """
    print(f"\nAnalyzing dataset: {file_path}")
    
    # Read data based on file type
    if file_path.endswith('.json'):
        print("Reading JSON file...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = []
        for category, entries in data.items():
            for person, person_texts in entries.items():
                texts.extend(person_texts)
    elif file_path.endswith('.txt'):
        print("Reading TXT file...")
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
    elif file_path.endswith('.csv'):
        print("Reading CSV file...")
        df = pd.read_csv(file_path)
        texts = df['text'].tolist()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    print(f"Found {len(texts)} texts to analyze")
    
    # Analyze texts
    results = []
    for i, text in enumerate(tqdm(texts, desc="Analyzing texts")):
        try:
            analysis = pipeline.analyze_text(text)
            
            # Extract relevant scores
            result = {
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
            results.append(result)
            
            # Save intermediate results
            if output_file and (i + 1) % batch_size == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(output_file, index=False)
                print(f"\nSaved intermediate results ({i + 1}/{len(texts)} texts processed)")
            
            # Rate limiting for API calls
            time.sleep(API_REQUEST_INTERVAL)
            
        except Exception as e:
            print(f"Error processing text {i + 1}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save final results if output file specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nSaved final results to {output_file}")
    
    return df

def main():
    # Initialize pipeline
    pipeline = TextAnalysisPipeline()
    
    # Example usage
    text = "The doctor performed the surgery with great skill and care."
    results = pipeline.analyze_text(text)
    
    print("\nExample Analysis Results:")
    print("=" * 50)
    print(f"Text: {text}")
    print("\nSentiment Analysis:")
    print(f"  Score: {results['sentiment']['scores']['compound']:.3f}")
    print(f"  Label: {results['sentiment']['sentiment']}")
    
    print("\nToxicity Analysis:")
    for attr, score in results['toxicity']['scores'].items():
        print(f"  {attr}: {score:.3f}")
    
    print("\nRegard Analysis:")
    print(f"  Label: {results['regard']['label']}")
    print(f"  Score: {results['regard']['score']:.3f}")
    
    print("\nGender Polarity Analysis:")
    for method in ['unigram', 'gender_max', 'gender_wavg']:
        print(f"  {method.upper()}:")
        print(f"    Score: {results['gender_polarity'][method]['score']:.3f}")
        print(f"    Label: {results['gender_polarity'][method]['label']}")

if __name__ == "__main__":
    main() 