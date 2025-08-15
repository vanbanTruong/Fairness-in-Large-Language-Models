# Datasets for Fairness and Bias Evaluation in Language Models

This is the artifact for the paper **[Datasets for Fairness in Language Models: An In-Depth Survey](https://arxiv.org/abs/2506.23411)**. This artifact aggregates and systematizes benchmark datasets used to evaluate fairness and social bias in language models (LMs). It provides a unified taxonomy and rich metadata describing each dataset’s structure, provenance, language coverage, bias types, and accessibility, together with reproducible code and standardized evaluation pipelines to support transparent, comparable fairness audits across models and tasks.

## Overview

This repository implements the dataset taxonomy, benchmarks, and evaluation pipelines described in the paper **[Datasets for Fairness in Language Models: An In-Depth Survey](https://arxiv.org/abs/2506.23411)**. It provides tools to reproduce the paper’s dataset curation, run standardized fairness analyses, and inspect dataset properties across tasks and languages.

## Research Contributions

This survey paper makes three key contributions to the field of language model fairness evaluation:

1. **Systematic Organization of Fairness Benchmarks**: A two-way taxonomy distinguishing between constrained-form and open-ended evaluation datasets, clarifying how dataset format shapes fairness findings.

2. **Unified Framework for Evaluating Dataset-Level Bias**: A comprehensive analysis pipeline enabling consistent, task-agnostic evaluation of dataset-level biases through standardized statistical estimators.

3. **Dataset-Specific Findings and Research Outlook**: Identification of both explicit and subtle forms of bias, revealing persistent identity-linked associations and highlighting key areas for advancement.

## Dataset Taxonomy

Our taxonomy organizes fairness datasets along two primary dimensions:

### 1. Structural Families

#### 📋 **Constrained-Form Evaluation Datasets**
Datasets requiring models to select or rank from predefined outputs, enabling clear, quantifiable bias measures.

**Coreference and Pronoun Resolution**
- **WinoBias**: Gender bias detection in coreference resolution with minimally contrasted sentence pairs
- **Winogender**: Gender-occupation association bias through template-based sentences
- **GAP**: Gender bias in coreference resolution with natural text examples

**Sentence-Likelihood Counterfactuals**
- **StereoSet**: Stereotype measurement through sentence pair comparisons
- **CrowS-Pairs**: Crowdsourced stereotype detection in natural language
- **RedditBias**: Social media bias analysis across multiple demographic dimensions
- **HolisticBias**: Comprehensive bias evaluation framework with controlled templates

**Classification-based Bias**
- **EEC (Equity Evaluation Corpus)**: Multi-dimensional fairness metrics for classification tasks
- **Bias-NLI**: Natural language inference bias detection and measurement

**Multiple-Choice Question Answering**
- **BBQ (Bias Benchmark for Question Answering)**: Expert-annotated bias evaluation across nine demographic categories
- **UnQover**: Question generation bias analysis through underspecified prompts

**Information Retrieval Bias**
- **Grep-BiasIR**: Search result bias detection and ranking fairness evaluation

#### 🚀 **Open-Ended Evaluation Datasets**
Datasets requiring models to generate free-form text, capturing emergent biases in unconstrained content generation.

- **BOLD**: Bias in open-ended language generation across multiple demographic dimensions
- **RealToxicityPrompts**: Toxicity detection and analysis in generated content
- **HONEST**: Hate speech and offensive content detection in multiple languages
- **TrustGPT**: Trustworthiness and safety evaluation in AI-generated responses

### 2. Attribute Dimensions

Each dataset is characterized along four orthogonal axes:

- **Source**: Template-based, natural text, crowdsourced, or AI-generated
- **Linguistic Coverage**: Monolingual (primarily English) vs. multilingual support
- **Bias Typology**: Demographic characteristics (gender, race, religion, etc.) and dataset construction factors
- **Accessibility**: Public availability and licensing terms

## Bias Analysis Framework

Our unified framework identifies and quantifies four types of dataset-level bias:

### 1. **Representativeness Bias** (Brep)
Measures divergence between dataset and population distributions using Kullback-Leibler divergence:
```
Brep = DKL(PD(A) || PP(A))
```

### 2. **Annotation Bias** (Bann)
Quantifies systematic differences in labeling outcomes across demographic groups:
```
Bann = max |E[gθ(x) | A = a1] - E[gθ(x) | A = a2]|
```

### 3. **Stereotype Leakage**
Uses information-theoretic measures (PMI and MI) to detect implicit associations between demographic descriptors and traits.

### 4. **Differential Metric Bias**
Identifies systematic performance disparities across protected attributes in evaluation metrics.

## Core Analysis Tools

### Gender Polarity Analyzer (`gender_polarity.py`)
Advanced gender bias detection using multiple methodologies:
- **Unigram Matching**: Count-based gender bias detection
- **Gender-Max**: Maximum gender polarization analysis
- **Gender-Wavg**: Weighted average gender polarity
- **Word Embedding Integration**: Support for various embedding models

### Text Analysis Pipeline (`text_analysis_pipeline.py`)
Comprehensive text analysis capabilities:
- **Sentiment Analysis**: VADER-based sentiment scoring
- **Toxicity Detection**: Google Perspective API integration
- **Regard Analysis**: BERT-based regard classification
- **Gender Bias Analysis**: Integrated gender polarity detection

### Dataset-Specific Analysis Scripts
Each dataset includes specialized analysis tools:
- **WinoBias**: `analyze_winobias_differential_matric_bias.py`, `analyze_winobias_representativeness_bias.py`
- **BBQ**: Multi-dimensional bias detection across demographic groups
- **TrustGPT**: Safety metrics and trust scoring with visualization tools

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/vanbanTruong/Fairness-in-Large-Language-Models.git
cd Fairness-in-Large-Language-Models/datasets

# Install required dependencies
pip install -r requirements.txt

# Download required NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage Examples

### Basic Gender Bias Analysis

```python
from gender_polarity import GenderPolarityAnalyzer

# Initialize analyzer
analyzer = GenderPolarityAnalyzer()

# Analyze text for gender bias
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

```bash
# WinoBias analysis
python constrained_form/WinoBias/analyze_winobias_differential_matric_bias.py

# BBQ bias evaluation
python constrained_form/BBQ/analyze_bbq_bias.py

# TrustGPT safety analysis
python open_ended/TrustGPT/analyze_trustgpt.py
```

## Project Structure

```
datasets/
├── constrained_form/           # Structured evaluation datasets
│   ├── WinoBias/              # Gender bias in coreference resolution
│   ├── Winogender/            # Gender-occupation associations
│   ├── GAP/                   # Coreference bias evaluation
│   ├── StereoSet/             # Stereotype measurement
│   ├── CrowS-Pairs/           # Crowdsourced stereotype detection
│   ├── RedditBias/            # Social media bias analysis
│   ├── HolisticBias/          # Comprehensive bias framework
│   ├── EEC/                   # Equity evaluation corpus
│   ├── Bias-NLI/              # Natural language inference bias
│   ├── BBQ/                   # Question answering bias benchmark
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

## Key Findings

Our analysis of 16 popular fairness datasets reveals:

- **Persistent Identity-Linked Associations**: Even under controlled conditions, models exhibit systematic biases
- **Annotation Inconsistencies**: Human and automated labeling functions show systematic demographic disparities
- **Representation Gaps**: Significant underrepresentation of marginalized groups in benchmark datasets
- **Cross-Dataset Patterns**: Consistent bias patterns emerge across different evaluation approaches

## Future Research Directions

The survey identifies critical areas for advancement:

1. **Multilingual Expansion**: Broader coverage beyond English-centric evaluations
2. **Intersectional Analysis**: Moving beyond single-attribute bias detection
3. **Community-Informed Governance**: Incorporating diverse perspectives in dataset design
4. **Low-Resource Settings**: Improving fairness metrics for underrepresented languages and cultures

## Contributing

We welcome contributions to improve bias detection methods and add support for new datasets. Please note that **all dataset analysis code is currently being continuously updated** to maintain accuracy and incorporate the latest research findings.


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

This project is licensed under the MIT License. See the LICENSE file in each dataset directory for specific licensing information.

---

**Note**: This toolkit is actively maintained with **continuous updates** to all dataset analysis code to ensure accuracy, performance, and compatibility with the latest research in bias detection and fairness evaluation. The unified taxonomy and analysis framework presented here enables researchers to conduct more systematic, comparable, and responsible fairness audits across language models and tasks.
