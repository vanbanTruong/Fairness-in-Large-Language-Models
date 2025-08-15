# Datasets for Fairness and Bias Evaluation in Language Models

A comprehensive toolkit for analyzing and evaluating fairness, bias, and representativeness in Large Language Models (LLMs) across multiple datasets and evaluation metrics.

## Overview

This project provides a robust framework for detecting and analyzing various types of bias in language models, including gender bias, racial bias, occupational bias, and other forms of social bias. It includes multiple analysis pipelines, visualization tools, and evaluation metrics to help researchers and practitioners understand and mitigate bias in their language models.

## Features

- **Multi-Dataset Support**: Comprehensive coverage of bias evaluation datasets
- **Multiple Analysis Methods**: Various bias detection and measurement techniques
- **Visualization Tools**: Interactive charts and graphs for bias analysis results
- **Pipeline Architecture**: Modular design for easy integration and extension
- **Gender Polarity Analysis**: Advanced gender bias detection using word embeddings
- **Sentiment and Toxicity Analysis**: Multi-dimensional text analysis capabilities

## Dataset Categories

The project organizes datasets into two main categories based on their evaluation approach:

### 📋 **Constrained Form Datasets**
These datasets use structured, constrained evaluation methods with predefined prompts, questions, or sentence pairs for bias detection.

#### 🔍 **WinoBias**
- **Purpose**: Gender bias detection in coreference resolution
- **Analysis**: Differential bias metrics, representativeness bias, statistical analysis
- **Files**: `analyze_winobias_differential_matric_bias.py`, `analyze_winobias_representativeness_bias.py`, `analyze_winobias_stats.py`

#### 🧠 **BBQ (Bias Benchmark for Question Answering)**
- **Purpose**: Bias evaluation in question-answering systems
- **Analysis**: Multi-dimensional bias detection across various demographic groups
- **Status**: Code continuously updated

#### 🎯 **CrowS-Pairs**
- **Purpose**: Stereotype detection in language models
- **Analysis**: Stereotype identification and measurement
- **Status**: Code continuously updated

#### 📊 **StereoSet**
- **Purpose**: Stereotype detection and measurement
- **Analysis**: Stereotype identification, bias quantification
- **Status**: Code continuously updated

#### 🔄 **UnQover**
- **Purpose**: Question generation bias analysis
- **Analysis**: Bias detection in question generation systems
- **Status**: Code continuously updated

#### 🌐 **HolisticBias**
- **Purpose**: Comprehensive bias evaluation framework
- **Analysis**: Multi-dimensional bias assessment
- **Status**: Code continuously updated

#### 🔍 **Grep-BiasIR**
- **Purpose**: Information retrieval bias analysis
- **Analysis**: Search result bias detection
- **Status**: Code continuously updated

#### 📝 **GAP**
- **Purpose**: Gender bias in coreference resolution
- **Analysis**: Gender-specific bias metrics
- **Status**: Code continuously updated

#### 🌍 **EEC (Equity Evaluation Corpus)**
- **Purpose**: Equity and fairness evaluation
- **Analysis**: Multi-dimensional fairness metrics
- **Status**: Code continuously updated

#### 🎯 **Bias-NLI**
- **Purpose**: Natural language inference bias detection
- **Analysis**: Inference bias measurement
- **Status**: Code continuously updated

#### 🎨 **BOLD**
- **Purpose**: Bias in open-ended language generation
- **Analysis**: Generation bias detection and measurement
- **Status**: Code continuously updated

#### 🧮 **RedditBias**
- **Purpose**: Social media bias analysis
- **Analysis**: Platform-specific bias detection
- **Status**: Code continuously updated

### 🚀 **Open-Ended Datasets**
These datasets evaluate bias through open-ended generation tasks, allowing models to produce free-form text responses.

#### 🎨 **BOLD**
- **Purpose**: Bias in open-ended language generation
- **Analysis**: Generation bias detection and measurement
- **Status**: Code continuously updated

#### ⚠️ **RealToxicityPrompts**
- **Purpose**: Toxicity detection and analysis
- **Analysis**: Toxicity scoring, prompt analysis, safety evaluation
- **Status**: Code continuously updated

#### 🎭 **HONEST**
- **Purpose**: Hate speech and offensive content detection
- **Analysis**: Content moderation, bias detection
- **Status**: Code continuously updated

#### 🤖 **TrustGPT**
- **Purpose**: Trustworthiness and safety evaluation
- **Analysis**: Safety metrics, trust scoring, visualization tools
- **Status**: Code continuously updated

## Core Analysis Tools

### Gender Polarity Analyzer (`gender_polarity.py`)
- **Unigram Matching**: Count-based gender bias detection
- **Gender-Max**: Maximum gender polarization analysis
- **Gender-Wavg**: Weighted average gender polarity
- **Word Embedding Integration**: Support for various embedding models

### Text Analysis Pipeline (`text_analysis_pipeline.py`)
- **Sentiment Analysis**: VADER-based sentiment scoring
- **Toxicity Detection**: Google Perspective API integration
- **Regard Analysis**: BERT-based regard classification
- **Gender Bias Analysis**: Integrated gender polarity detection

### Analysis Utilities
- **Visualization Tools**: Matplotlib-based chart generation
- **Data Export**: CSV and JSON output formats
- **Batch Processing**: Efficient handling of large datasets
- **Compatibility Fixes**: Cross-platform compatibility solutions

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Fairness-in-Large-Language-Models

# Install required dependencies
pip install -r requirements.txt

# Download required NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

### Basic Gender Bias Analysis

```python
from gender_polarity import GenderPolarityAnalyzer

# Initialize analyzer
analyzer = GenderPolarityAnalyzer()

# Analyze text
text = "The doctor went to his office."
results = analyzer.analyze_text(text)
print(results)
```

### Comprehensive Text Analysis

```python
from text_analysis_pipeline import TextAnalysisPipeline

# Initialize pipeline
pipeline = TextAnalysisPipeline()

# Analyze text comprehensively
text = "Sample text for analysis"
results = pipeline.analyze_text(text)
print(results)
```

### Dataset-Specific Analysis

```python
# WinoBias analysis
python constrained_form/WinoBias/analyze_winobias_differential_matric_bias.py

# BBQ analysis
python constrained_form/BBQ/analyze_bbq_bias.py

# TrustGPT analysis
python open_ended/TrustGPT/analyze_trustgpt.py
```

## Project Structure

```
Fairness-in-Large-Language-Models/
├── constrained_form/           # Structured evaluation datasets
│   ├── WinoBias/              # Gender bias analysis tools
│   ├── Winogender/            # Gender bias detection
│   ├── GAP/                   # Coreference bias
│   ├── StereoSet/             # Stereotype measurement
│   ├── CrowS-Pairs/           # Stereotype detection
│   ├── RedditBias/            # Social media bias
│   ├── HolisticBias/          # Comprehensive bias framework
│   ├── EEC/                   # Equity evaluation
│   ├── Bias-NLI/              # Natural language inference bias
│   ├── BBQ/                   # Question answering bias evaluation
│   ├── UnQover/               # Question generation bias
│   ├── Grep-BiasIR/           # Information retrieval bias
│   └── BOLD/                  # Generation bias (constrained)
├── open_ended/                 # Open-ended generation datasets
│   ├── BOLD/                  # Generation bias (open-ended)
│   ├── RealToxicityPrompts/   # Toxicity analysis
│   ├── HONEST/                # Hate speech detection
│   └── TrustGPT/              # Trustworthiness evaluation
├── gender_polarity.py          # Core gender bias analyzer
├── text_analysis_pipeline.py   # Main analysis pipeline
├── compat_fix.py               # Compatibility utilities
└── README.md                   # This file
```

## Contributing

We welcome contributions to improve bias detection methods and add support for new datasets. Please note that **all dataset analysis code is continuously updated** to maintain accuracy and incorporate the latest research findings.

### Areas for Contribution
- New bias detection algorithms
- Additional dataset support
- Improved visualization tools
- Performance optimizations
- Documentation improvements

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@article{zhang2025datasets,
  title={Datasets for Fairness in Language Models: An In-Depth Survey},
  author={Zhang, Jiale and Wang, Zichong and Palikhe, Avash and Yin, Zhipeng and Zhang, Wenbin},
  journal={arXiv preprint arXiv:2506.23411},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file in each dataset directory for specific licensing information.

## Acknowledgments

- Dataset creators and maintainers
- Research community contributions
- Open-source bias detection tools
- Academic institutions supporting fairness research

## Contact

For questions, suggestions, or contributions, please open an issue on GitHub or contact the maintainers.

---

**Note**: This toolkit is actively maintained with **continuous updates** to all dataset analysis code to ensure accuracy, performance, and compatibility with the latest research in bias detection and fairness evaluation.
