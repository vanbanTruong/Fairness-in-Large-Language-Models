# Fairness Definitions in Language Models Explained

This is the artifact for the paper [Fairness Definitions in Language Models Explained](). This artifact supplies the tools and implementation guidelines to reproduce and apply fairness definitions discussed in our paper. 


## Installation

Install required packages/libraries:

```shell script
$ pip install -r requirements.txt
```
For [GPT](https://openai.com/api/) and [Llama](https://www.together.ai/), access keys are required for API requests. Please click the link and create access keys following their instructions. After obtaining the access credentials, fill them in  `api_key.py`.

```shell script
OPENAI_KEY = "your openai key" # for OpenAI GPT
TOGETHERAPI_KEY = "your togetherapi key" # for Llama2 
```

## Run Experiments

This section is organized according to the section in our paper. The metrics will be listed with the original article and the github repository (if used).  

### Fairness definitions for encoder-only language models

**Intrinsic bias** 

* Similarity-based disparity: To run experiment test for similarity-based disparity, run the script with one of the following `<metric_name>`
  
  * **weat**: Semantics derived automatically from language corpora contain human-like biases [[arXiv]](https://arxiv.org/abs/1608.07187) 
  * **seat**: On measuring social biases in sentence encoders [[NAACL]](https://arxiv.org/abs/1903.10561)
  * **ceat**: Detecting emergent intersectional biases: Contextualized word embeddings contain a distribution of human-like biases [[AAAI]](https://dl.acm.org/doi/abs/10.1145/3461702.3462536)
  
```shell script
$ python main.py encoder-only intrinsic similarity-based-disparity <metric_name>
```

* Probability-based disparity: To run experiment test for probability-based disparity, run the script with one of the following `<metric_name>`
  
  * **disco**: Measuring and reducing gendered correlations in pre-trained models [[arXiv]](https://arxiv.org/abs/2010.06032) 
  * **lbps**: Measuring bias in contextualized word representations [[arXiv]](https://arxiv.org/abs/1906.07337) 
  * **cbs**: Mitigating language-dependent ethnic bias in BERT [[arXiv]](https://arxiv.org/abs/2109.05704) 
  * **pll**: Masked language model scoring [[arXiv]](https://arxiv.org/abs/1910.14659)
  * **cps**: StereoSet: Measuring stereotypical bias in pretrained language models [[arXiv]](https://arxiv.org/abs/2004.09456)
  * **aul**: Unmasking the mask–evaluating social biases in masked language models [[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/download/21453/21202)
  * **aula**: Unmasking the mask–evaluating social biases in masked language models [[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/download/21453/21202)
  * **cat**: CrowS-pairs: A challenge dataset for measuring social biases in masked language models [[arXiv]](https://arxiv.org/abs/2010.00133)
    
```shell script
$ python main.py encoder-only intrinsic probability-based-disparity <metric_name>
```
  
**Extrinsic bias** 

* Equal opportunity: To run experiment test for equal opportunity, run the script with one of the following `<metric_name>`
  
  * **gap**: Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting [[ACM]](https://dl.acm.org/doi/abs/10.1145/3287560.3287572)

```shell script
$ python main.py encoder-only extrinsic equal-opportunity <metric_name>
```

* Fair inference: To run experiment test for fair inference, run the script with one of the following `<metric_name>`
  
  * **nn**, **fn**, **t_0.5**, **t_0.7**: On Measuring and Mitigating Biased Inferences of Word Embeddings [[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/view/6267)

```shell script
$ python main.py encoder-only extrinsic fair-inference <metric_name>
```

* Context-based disparity: To run experiment test for context-based disparity, run the script with one of the following `<metric_name>`
  
  * **s_amb**, **s_dis**: BBQ: A hand-built bias benchmark for question answering [[ACL]](https://aclanthology.org/2022.findings-acl.165)

```shell script
$ python main.py encoder-only extrinsic context-based-disparity <metric_name>
```

### Fairness definitions for decoder-only language models

**Intrinsic bias** 

* Attention head-based disparity: To run experiment test for attention head-based disparity, run the script with one of the following `<metric_name>`
  
  * **nie**: Investigating Gender Bias in Language Models Using Causal Mediation Analysis [[NeurIPS]](https://proceedings.neurips.cc/paper/2020/file/92650b2e92217715fe312e6fa7b90d82-Paper.pdf)
  * **gbe**: Bias A-head? Analyzing Bias in Transformer-Based Language Model Attention Heads [[ACL]](https://aclanthology.org/2025.trustnlp-main.18) 
  
```shell script
$ python main.py decoder-only intrinsic attention-head-based-disparity <metric_name>
```

* Stereotypical Association: To run experiment test for stereotypical association, run the script with one of the following `<metric_name>`
  
  * **sll**: Language models are few-shot learners [[ACM]](https://dl.acm.org/doi/abs/10.5555/3495724.3495883) 
  * **ca**: Holistic Evaluation of Language Models [[arXiv]](https://arxiv.org/abs/2211.09110) 
    
```shell script
$ python main.py decoder-only intrinsic stereotypical-association <metric_name>
```
  
**Extrinsic bias** 

* Counterfactual fairness: To run experiment test for counterfactual fairness, run the script with one of the following `<metric_name>`
  
  * **cr**: Fairness of ChatGPT [[arXiv]](https://arxiv.org/abs/2305.18569)
  * **ctf**: Improving Counterfactual Generation for Fair Hate Speech Detection [[ACL]](https://aclanthology.org/2021.woah-1.10)

```shell script
$ python main.py decoder-only extrinsic counterfactual-fairness <metric_name>
```

* Performance disparity: To run experiment test for performance disparity, run the script with one of the following `<metric_name>`
  
  * **ad**: Holistic Evaluation of Language Models [[arXiv]](https://arxiv.org/abs/2211.09110)
  * **ba**: BiasAsker: Measuring the Bias in Conversational AI System [[ACM]](https://dl.acm.org/doi/abs/10.1145/3611643.3616310)
  * **sns**: Is chatgpt fair for recommendation? evaluating fairness in large language model recommendation [[arXiv]](https://arxiv.org/abs/2305.07609)

```shell script
$ python main.py decoder-only extrinsic performance-disparity <metric_name>
```

* Demographic representation representation: To run experiment test for demographic representation, run the script with one of the following `<metric_name>`
  
  * **drd**: Holistic Evaluation of Language Models [[arXiv]](https://arxiv.org/abs/2211.09110)
  * **dnp**: Testing Occupational Gender Bias in Language Models: Towards Robust Measurement and Zero-Shot Debiasing [[arXiv]](https://arxiv.org/abs/2212.10678v2)

```shell script
$ python main.py decoder-only extrinsic demographic-representation <metric_name>
```

### Fairness definitions for encoder-decoder language models

**Intrinsic bias** 

* Algorithmic disparity: To run experiment test for algorithmic disparity, run the script with one of the following `<metric_name>`
  
  * **lfp**, **mcd**: Machine Translationese: Effects of Algorithmic Bias on Linguistic Complexity in Machine Translation [[ACL]](https://aclanthology.org/2021.eacl-main.188)

```shell script
$ python main.py encoder-decoder intrinsic algorithmic-disparity <metric_name>
```

* Stereotypical association: To run experiment test for stereotypical association, run the script with one of the following `<metric_name>`
  
  * **sd**: A Tale of Pronouns: Interpretability Informs Gender Bias Mitigation for Fairer Instruction-Tuned Machine Translation [[ACL]](https://aclanthology.org/2023.emnlp-main.243)
  * **sva**: Deciphering Stereotypes in Pre-Trained Language Models [[ACL]](https://aclanthology.org/2023.emnlp-main.697/)

```shell script
$ python main.py encoder-decoder intrinsic stereotypical-association <metric_name>
```
  
**Extrinsic bias** 

* Position-based disparity: To run experiment test for position-based disparity, run the script with one of the following `<metric_name>`
  
  * **npd**: Revisiting Zero-Shot Abstractive Summarization in the Era of Large Language Models from the Perspective of Position Bias [[arXiv]](https://arxiv.org/abs/2401.01989)

```shell script
$ python main.py encoder-decoder extrinsic position-based-disparity <metric_name>
```

* Fair inference: To run experiment test for fair inference, run the script with one of the following `<metric_name>`
  
  * **ibs**: On Measuring Social Biases in Prompt-Based Multi-Task Learning [[ACL]](https://aclanthology.org/2022.findings-naacl.42)

```shell script
$ python main.py encoder-decoder extrinsic fair-inference <metric_name>
```

* Individual fairness: To run experiment test for individual fairness, run the script with one of the following `<metric_name>`
  
  * **ss**: Fairness Testing of Machine Translation Systems [[ACM]](https://dl.acm.org/doi/10.1145/3664608)

```shell script
$ python main.py encoder-decoder extrinsic individual-fairness <metric_name>
```

* Counterfactual fairness: To run experiment test for counterfactual fairness, run the script with one of the following `<metric_name>`
  
  * **auc**: UP5: Unbiased Foundation Model for Fairness-aware Recommendation [[arXiv]](https://arxiv.org/abs/2305.12090)

```shell script
$ python main.py encoder-decoder extrinsic counterfactual-fairness <metric_name>
```

<!-- ### Fairness definitions for medium-sized language models

**Intrinsic bias** 

* Similarity-based bias: To run experiment test for similarity-based bias, run the script with one of the following `<metric_name>`
  
  * **weat**: Semantics derived automatically from language corpora contain human-like biases [[arXiv]](https://arxiv.org/abs/1608.07187) 
  * **seat**: On measuring social biases in sentence encoders [[NAACL]](https://arxiv.org/abs/1903.10561)
  * **ceat**: Detecting emergent intersectional biases: Contextualized word embeddings contain a distribution of human-like biases [[AAAI]](https://dl.acm.org/doi/abs/10.1145/3461702.3462536)
  
```shell script
$ python main.py medium intrinsic similarity <metric_name>
```

* Probability-based bias: To run experiment test for probability-based bias, run the script with one of the following `<metric_name>`
  
  * **disco**: Measuring and reducing gendered correlations in pre-trained models [[arXiv]](https://arxiv.org/abs/2010.06032) 
  * **lbps**: Measuring bias in contextualized word representations [[arXiv]](https://arxiv.org/abs/1906.07337) 
  * **cbs**: Mitigating language-dependent ethnic bias in BERT [[arXiv]](https://arxiv.org/abs/2109.05704) 
  * **ppl**: Masked language model scoring [[arXiv]](https://arxiv.org/abs/1910.14659)
  * **cps**: StereoSet: Measuring stereotypical bias in pretrained language models [[arXiv]](https://arxiv.org/abs/2004.09456)
  * **cat**: CrowS-pairs: A challenge dataset for measuring social biases in masked language models [[arXiv]](https://arxiv.org/abs/2010.00133)
  * **aul**: Unmasking the mask–evaluating social biases in masked language models [[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/download/21453/21202)
  * **aula**: Unmasking the mask–evaluating social biases in masked language models [[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/download/21453/21202)
    
```shell script
$ python main.py medium intrinsic probability <metric_name>
```

**Extrinsic bias**

* Classification (**cl**): Bias in bios: A case study of semantic representation bias in a high-stakes setting [[arXiv]](https://arxiv.org/pdf/1901.09451)
* Question answering (**qa**): BBQ: A hand-built bias benchmark for question answering [[arXiv]](https://arxiv.org/pdf/2110.08193)

```shell script
$ python main.py medium extrinsic <task_name>
```

### Fairness definitions for large-sized language models

* Demographic Representation (**dr**)
  * **exp1**: Language models are few-shot learners [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
  * **exp2**: Understanding stereotypes in language models: Towards robust measurement and zero-shot debiasing [[arXiv]](https://arxiv.org/pdf/2212.10678)
  * **exp3**: Holistic evaluation of language models [[arXiv]](https://arxiv.org/pdf/2211.09110)
    
* Stereotypical Association (**sa**)
  * **exp1**: Language models are few-shot learners [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
  * **exp2**: Holistic evaluation of language models [[arXiv]](https://arxiv.org/pdf/2211.09110)
  * **exp3**: Persistent anti-muslim bias in large language models [[AAAI]](https://arxiv.org/pdf/2101.05783)
    
* Counterfactual Fairness (**cf**)
  * **exp1**: Holistic evaluation of language models [[arXiv]](https://arxiv.org/pdf/2211.09110)
  * **exp2**: Fairness of chatgpt [[arXiv]](https://arxiv.org/pdf/2305.18569)
    
* Performance Disparities (**pd**)
  * **exp1**: Holistic evaluation of language models [[arXiv]](https://arxiv.org/pdf/2211.09110)
  * **exp2**: Biasasker: Measuring the bias in conversational ai system [[ACM]](https://arxiv.org/pdf/2305.12434)
  * **exp3**: Is chatgpt fair for recommendation? evaluating fairness in large language model recommendation [[ACM]](https://arxiv.org/pdf/2305.07609)
    
```shell script
$ python main.py large <strategy_name> <experiment_name>
``` -->
