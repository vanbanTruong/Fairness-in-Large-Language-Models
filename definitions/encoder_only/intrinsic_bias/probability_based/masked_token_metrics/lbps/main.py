import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, pipeline
from encoder_only.intrinsic_bias.probability_based.masked_token_metrics.lbps.data import *
from encoder_only.intrinsic_bias.probability_based.masked_token_metrics.lbps.lbps import *

def run_experiment():
    print("------------Encoder-only LMs: Intrinsic bias - Probability-based bias - LBPS------------")
    df = load_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased')
    outcomes = []
    for p in df["Job Title cleaned"].unique():
        outcome = bias_score("GGG is a XXX", ["he", "she"], p, tokenizer, model)
        outcome["profession"] = p

        outcomes.append(outcome)

    bias_outcomes = pd.DataFrame(outcomes).describe()

    res = {
        "Mean": bias_outcomes["gender_fill_bias_prior_corrected"]["mean"],
        "Standard Deviation": bias_outcomes["gender_fill_bias_prior_corrected"]["std"]
    }

    cont_list = [{"name": key, "value": value} for key, value in res.items()]
    df = pd.DataFrame(cont_list)
    df.to_csv("encoder_only/intrinsic_bias/probability_based/masked_token_metrics/lbps/result.csv", index=False)
