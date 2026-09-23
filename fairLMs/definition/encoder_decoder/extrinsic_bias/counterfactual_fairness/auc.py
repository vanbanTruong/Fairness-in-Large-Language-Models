import re
from typing import List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_MALE_TERMS   = ["he", "him", "his", "himself"]
_FEMALE_TERMS = ["she", "her", "hers", "herself"]

_M2F = {"he": "she", "him": "her", "his": "her", "himself": "herself"}
_F2M = {"she": "he",  "her": "him", "hers": "his", "herself": "himself"}

_NAT_A2B = {
    "american":  "chinese",
    "european":  "african",
    "western":   "eastern",
    "white":     "black",
    "christian": "muslim",
    "english":   "arabic",
    "french":    "arabic",
    "german":    "arabic",
}
_NAT_B2A = {v: k for k, v in _NAT_A2B.items()}
_NAT_ALL  = {**_NAT_A2B, **_NAT_B2A}

_NAT_GROUP_A = set(_NAT_A2B.keys())
_NAT_GROUP_B = set(_NAT_A2B.values())


def _contains_any(text: str, terms: List[str]) -> bool:
    lowered = text.lower()
    for term in terms:
        if re.search(r"\b" + re.escape(term) + r"\b", lowered):
            return True
    return False


def _swap_gender(text: str) -> Tuple[str, bool]:
    has_m = _contains_any(text, _MALE_TERMS)
    has_f = _contains_any(text, _FEMALE_TERMS)
    if has_m and not has_f:
        table = _M2F
    elif has_f and not has_m:
        table = _F2M
    else:
        return text, False

    swapped = re.sub(
        r"\b(" + "|".join(re.escape(k) for k in table) + r")\b",
        lambda m: table[m.group().lower()],
        text, flags=re.IGNORECASE,
    )
    return swapped, swapped != text


def _swap_nationality(text: str) -> Tuple[str, bool]:
    has_a = _contains_any(text, list(_NAT_GROUP_A))
    has_b = _contains_any(text, list(_NAT_GROUP_B))
    if has_a == has_b:
        return text, False

    swapped = re.sub(
        r"\b(" + "|".join(re.escape(k) for k in _NAT_ALL) + r")\b",
        lambda m: _NAT_ALL[m.group().lower()],
        text, flags=re.IGNORECASE,
    )
    return swapped, swapped != text


@torch.no_grad()
def _get_encoder_embedding(
    model:      AutoModelForSeq2SeqLM,
    tokenizer:  AutoTokenizer,
    text:       str,
    max_length: int = 128,
) -> np.ndarray:
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(model.device)

    encoder     = model.get_encoder()
    encoder_out = encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    hidden = encoder_out.last_hidden_state
    mask   = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled.squeeze(0).cpu().float().numpy()


def _auc_one_split(X, y, groups, seed, test_ratio):
    rng = np.random.default_rng(seed)
    if groups is not None:
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        n_test_pairs = max(1, int(round(len(uniq) * test_ratio)))
        test_pairs   = set(uniq[:n_test_pairs].tolist())
        test_mask    = np.isin(groups, list(test_pairs))
        test  = np.where(test_mask)[0]
        train = np.where(~test_mask)[0]
    else:
        idx   = rng.permutation(len(y))
        split = int(len(y) * (1 - test_ratio))
        train, test = idx[:split], idx[split:]

    if len(train) < 2 or len(test) < 1 or len(np.unique(y[test])) < 2:
        return None

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X[train])
    X_test  = scaler.transform(X[test])
    clf = LogisticRegression(max_iter=500, solver="lbfgs", random_state=int(seed))
    clf.fit(X_train, y[train])
    proba = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y[test], proba))


def compute_auc(
    model:      AutoModelForSeq2SeqLM,
    tokenizer:  AutoTokenizer,
    sentences:  List[str],
    labels:     List[int],
    pair_ids:   List[int] = None,
    test_ratio: float = 0.2,
    seed:       int   = 42,
    n_seeds:    int   = 10,
) -> Tuple[float, float, int, int, list]:
    assert len(sentences) == len(labels), "sentences and labels must match"

    embeddings: List[np.ndarray] = []
    rows:       list             = []

    for i, (sentence, label) in enumerate(zip(sentences, labels)):
        emb = _get_encoder_embedding(model, tokenizer, sentence)
        embeddings.append(emb)
        rows.append({
            "index":    i,
            "pair_id":  pair_ids[i] if pair_ids is not None else i,
            "input":    sentence[:80],
            "label":    label,
            "emb_norm": float(np.linalg.norm(emb)),
        })

    y  = np.array(labels)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 < 2 or n1 < 2:
        return 0.0, 0.0, n0, n1, rows

    X = np.stack(embeddings)
    groups = np.asarray(pair_ids) if pair_ids is not None else None

    aucs = []
    for s in range(n_seeds):
        a = _auc_one_split(X, y, groups, seed + s, test_ratio)
        if a is not None:
            aucs.append(a)

    if not aucs:
        return 0.0, 0.0, n0, n1, rows

    auc_mean = round(float(np.mean(aucs)), 4)
    auc_std  = round(float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0, 4)
    return auc_mean, auc_std, n0, n1, rows