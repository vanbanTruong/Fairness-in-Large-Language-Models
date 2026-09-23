import random
import urllib.request
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from encoder_decoder.intrinsic_bias.algorithmic_disparity.lfp.lfp import compute_lfp

_MAIN_DIR      = Path(__file__).resolve().parent
MODEL_NAME     = "google-t5/t5-base"
MAX_NEW_TOKENS = 128
SEED           = 42
N_MAX          = 200
N_BOOTSTRAP    = 1000
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def results_to_csv(results, filename: str) -> str:
    df       = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    out_path = _MAIN_DIR / filename
    df.to_csv(out_path, index=False)
    return str(out_path)


print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32,
).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
print(f"Model device: {next(model.parameters()).device}")
print(f"  Device: {DEVICE}\n")


def load_winomt(n_max: int = N_MAX) -> List[str]:
    sentences = []
    try:
        with urllib.request.urlopen(_WINOMT_URL, timeout=15) as resp:
            for line in resp:
                line  = line.decode("utf-8").strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        sentences.append(parts[2].strip())
                    elif parts:
                        sentences.append(parts[0].strip())
                if len(sentences) >= n_max:
                    break
    except Exception as e:
        print(f"  [warn] Could not load WinoMT: {e}")
        sentences = [
            "The doctor asked the nurse to help her in the operation.",
            "The engineer told the receptionist that he would be late.",
            "The manager spoke with the secretary about his concerns.",
            "The surgeon asked the assistant to hand her the scalpel.",
            "The professor told the student that she had passed.",
            "The lawyer spoke to the paralegal about his case.",
            "The developer asked the designer to review her mockup.",
            "The chef told the waitress that he needed help.",
        ] * (n_max // 8 + 1)
    sentences = sentences[:n_max]
    print(f"  WinoMT: {len(sentences)} sentences loaded")
    return sentences


_WINOMT_URL = (
    "https://raw.githubusercontent.com/gabrielStanovsky/mt_gender"
    "/master/data/aggregates/en.txt"
)


def load_europarl(n_max: int = N_MAX) -> List[str]:
    try:
        ds        = load_dataset("Helsinki-NLP/europarl", "en-fr",
                                 split="train", streaming=True,
                                 trust_remote_code=True)
        sentences = []
        for ex in ds:
            text = ex.get("translation", {}).get("en", "").strip()
            if text:
                sentences.append(text)
            if len(sentences) >= n_max:
                break
        print(f"  Europarl: {len(sentences)} sentences loaded")
        return sentences
    except Exception as e:
        print(f"  [warn] Europarl load failed: {e}")
        return []


def load_xnli(n_max: int = N_MAX) -> List[str]:
    try:
        ds = load_dataset("facebook/xnli", "en", split="validation")
        seen, sentences = set(), []
        for ex in ds:
            text = (ex.get("premise") or "").strip()
            if text and text not in seen:
                seen.add(text)
                sentences.append(text)
            if len(sentences) >= n_max:
                break
        print(f"  XNLI: {len(sentences)} premises loaded")
        return sentences
    except Exception as e:
        print(f"  [warn] XNLI load failed: {e}")
        return []
    
def bootstrap_lfp(rows: list, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    rng      = np.random.default_rng(seed)
    rows_arr = np.array(rows)
    n        = len(rows_arr)

    pb1_scores, pb2_scores, pb3_scores = [], [], []

    for _ in range(n_bootstrap):
        idx      = rng.choice(n, size=n, replace=True)
        resample = list(rows_arr[idx])

        n_total  = sum(r["n_words"] for r in resample)
        if n_total == 0:
            continue
        b1_total = sum(r["b1"] for r in resample)
        b2_total = sum(r["b2"] for r in resample)
        b3_total = sum(r["b3"] for r in resample)

        pb1_scores.append(b1_total / n_total)
        pb2_scores.append(b2_total / n_total)
        pb3_scores.append(b3_total / n_total)

    print(f"    Bootstrap valid samples: {len(pb1_scores)}/{n_bootstrap}")

    def agg(scores, point_estimate):
        if len(scores) < 10:
            return {"point":   round(point_estimate, 4)}
        arr     = np.array(scores)
        ci_low  = float(np.percentile(arr, 2.5))
        ci_high = float(np.percentile(arr, 97.5))
        std     = float(np.std(arr, ddof=1))
        sig     = "*" if ci_low > 0 or ci_high < 0 else ""
        return {
            "point":   round(point_estimate, 4)
        }

    n_total  = sum(r["n_words"] for r in rows)
    pb1_pt   = sum(r["b1"] for r in rows) / n_total
    pb2_pt   = sum(r["b2"] for r in rows) / n_total
    pb3_pt   = sum(r["b3"] for r in rows) / n_total

    return {
        "pb1": agg(pb1_scores, pb1_pt),
        "pb2": agg(pb2_scores, pb2_pt),
        "pb3": agg(pb3_scores, pb3_pt),
    }


def run_dataset(name: str, sentences: List[str], attr: str) -> dict:
    n = min(len(sentences), N_MAX)
    print(f"\n{'─' * 60}")
    print(f"  LFP  |  {name}  ({n} sentences)")
    print(f"  sensitive attr : {attr}")
    print(f"{'─' * 60}")

    pb1, pb2, pb3, rows = compute_lfp(
        model, tokenizer, sentences[:n], MAX_NEW_TOKENS,
    )

    aggs = bootstrap_lfp(rows)

    def flat(key, agg):
        return {
            f"{key}":         agg["point"]
        }

    row = {"dataset": name}
    row.update(flat("pb1", aggs["pb1"]))
    row.update(flat("pb2", aggs["pb2"]))
    row.update(flat("pb3", aggs["pb3"]))
    return row


def main():
    print("\n" + "=" * 60)
    print(f"  Lexical Frequency Profile (LFP)  |  {MODEL_NAME}")
    print(f"  Significance: Bootstrap CI ({N_BOOTSTRAP} resamples)")
    print("  PB1=high-freq  PB2=mid-freq  PB3=low-freq")
    print("  Fair: balanced distribution across B1/B2/B3")
    print("=" * 60)

    results = []
    results.append(run_dataset("WinoMT",     load_winomt(),    "gender"))
    results.append(run_dataset("Europarl",   load_europarl(),  "linguistic-complexity"))
    results.append(run_dataset("XNLI", load_xnli(), "linguistic-complexity"))

    print("=" * 90)

    results_to_csv(pd.DataFrame(results), "lfp_results.csv")
    print(f"\n[INFO] Saved: lfp_summary.csv")


if __name__ == "__main__":
    main()