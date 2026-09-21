import csv
import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_ind
import torch
from transformers import BertTokenizer, BertModel
from fairLLMs.definition.encoder_only.intrinsic_bias.similarity_based.weat.data import ALL_TESTS
from fairLLMs.definition.encoder_only.intrinsic_bias.similarity_based.weat.weat import compute_weat
from pathlib import Path


_MAIN_DIR = os.path.dirname(__file__)


def results_to_csv(results: list, filename: str) -> str:
    out_path = os.path.join(_MAIN_DIR, filename)
    pd.DataFrame(results).to_csv(out_path, index=False)
    return str(out_path)

def load_bert(model_name="bert-base-uncased"):
    """Load tokeniser and model; return both on the appropriate device."""
    print("Loading")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded '{model_name}' on {device}")
    return tokenizer, model, device


@torch.no_grad()
def get_embedding(word, tokenizer, model, device):
    inputs = tokenizer(word.lower(), return_tensors="pt").to(device)
    outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state[0, 1:-1, :]
    embedding = token_embeddings.mean(dim=0).cpu().numpy()
    return embedding


def build_embedding_dict(words, tokenizer, model, device):
    return {w: get_embedding(w, tokenizer, model, device) for w in words}


def run_weat(test, tokenizer, model, device):
    print(f"\n  Building embeddings for test: {test['name']}")
    np.random.seed(43)
    all_words = test["t1"] + test["t2"] + test["a1"] + test["a2"]
    emb = build_embedding_dict(all_words, tokenizer, model, device)
    t1_vecs = [emb[w] for w in test["t1"]]
    t2_vecs = [emb[w] for w in test["t2"]]
    a1_vecs = [emb[w] for w in test["a1"]]
    a2_vecs = [emb[w] for w in test["a2"]]
    d, p = compute_weat(t1_vecs, t2_vecs, a1_vecs, a2_vecs)
    return {"test": test["name"], "effect_size_d": round(d, 4), "p_value": round(p, 4)}


BIAS_TYPE_MAP = {
    "C1": "race",
    "C2": "gender",
    "C3": "disease",
    "C4": "age",
}


def main():
    model_name = "bert-base-uncased"
    tokenizer, model, device = load_bert(model_name)

    print("\n" + "=" * 60)
    print("  WEAT on BERT-base-uncased  |  Caliskan et al. datasets")
    print("=" * 60)

    results = []
    for test in ALL_TESTS:
        r = run_weat(test, tokenizer, model, device)
        results.append(r)

    print("\n" + "=" * 60)
    print(f"{'Test':<55} {'d':>8}  {'p':>8}")
    print("-" * 60)
    for r in results:
        sig = "*" if r["p_value"] < 0.05 else ""
        print(f"{r['test']:<55} {r['effect_size_d']:>8.4f}  {r['p_value']:>7.4f}{sig}")
    print("=" * 60)

    results_to_csv(results, "weat_results.csv")


if __name__ == "__main__":
    main()