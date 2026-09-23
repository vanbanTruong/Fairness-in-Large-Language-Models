import csv
import pandas as pd
import torch
import os
from transformers import BertTokenizer, BertModel
from encoder_only.intrinsic_bias.similarity_based.weat.data import ALL_TESTS
from encoder_only.intrinsic_bias.similarity_based.seat.seat import compute_seat
from pathlib import Path
import numpy as np

# Maps test name prefix to bias type
BIAS_TYPE_MAP = {
    "C1": "race",
    "C2": "gender",
    "C3": "disease",
    "C4": "age",
}


_MAIN_DIR = Path(__file__).resolve().parent


def results_to_csv(results: list, filename: str) -> str:
    out_path = os.path.join(_MAIN_DIR, filename)
    pd.DataFrame(results).to_csv(out_path, index=False)
    return str(out_path)

def load_bert(model_name="bert-base-uncased"):
    """Load BERT tokeniser and model; return both on the appropriate device."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded '{model_name}' on {device}")
    return tokenizer, model, device


def run_seat(test, model, tokenizer, device, pooling = "mean"):
    """Run one SEAT test; return a results dict."""
    np.random.seed(43)
    d, p = compute_seat(
        model=model,
        tokenizer=tokenizer,
        T1_terms=test["t1"],
        T2_terms=test["t2"],
        A1_terms=test["a1"],
        A2_terms=test["a2"],
        pooling=pooling,
        n_samples=10_000,
        device=device,
    )
    return {
        "test":          test["name"],
        "effect_size_d": round(d, 4),
        "p_value":       round(p, 4),
    }



def main():
    tokenizer, model, device = load_bert()

    print("\n" + "=" * 60)
    print("  SEAT on BERT-base-uncased  |  Caliskan et al. datasets")
    print("=" * 60)

    results = []
    for test in ALL_TESTS:
        r = run_seat(test, model, tokenizer, device, "mean")
        results.append(r)

    print("\n" + "=" * 60)
    print(f"{'Test':<55} {'d':>8}  {'p':>8}")
    print("-" * 60)
    for r in results:
        sig = "*" if r["p_value"] < 0.05 else ""
        print(f"{r['test']:<55} {r['effect_size_d']:>8.4f}  {r['p_value']:>7.4f}{sig}")
    print("=" * 60)
    print()

    results_to_csv(results, "seat_results.csv")


if __name__ == "__main__":
    main()