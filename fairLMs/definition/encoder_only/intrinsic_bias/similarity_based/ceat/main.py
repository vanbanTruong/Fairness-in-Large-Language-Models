"""Demo of ``CEAT`` using ALL_TESTS + Reddit contexts (or template fallback)."""

from __future__ import annotations

import pickle
from pathlib import Path

from fairLMs.definition.encoder_only.intrinsic_bias.similarity_based.ceat.data import ALL_TESTS
from fairLMs.metrics import CEAT
from fairLMs.models import HuggingFaceModel

_DIR = Path(__file__).resolve().parent
SEN_DICT = _DIR / "sen_dic_1.pickle"

# Minimal templates used when Reddit pickle contexts are unavailable for a term.
_FALLBACK_TEMPLATES = [
    "The word {w} appears in this sentence.",
    "People often mention {w} in conversation.",
    "This passage is about {w} and related ideas.",
    "Consider the meaning of {w} carefully.",
    "Researchers studied {w} in detail.",
    "A story involving {w} was published recently.",
    "Discussion of {w} continued for hours.",
    "Everyone remembered {w} from the meeting.",
]


def _contexts_for_terms(terms, sen_dic, n_ctx=8):
    out = {}
    for w in terms:
        sents = []
        if sen_dic is not None:
            raw = sen_dic.get(w) or sen_dic.get(w.lower()) or []
            sents = [s for s in raw if isinstance(s, str) and w.lower() in s.lower()][:n_ctx]
        if len(sents) < n_ctx:
            sents = (_FALLBACK_TEMPLATES * ((n_ctx // len(_FALLBACK_TEMPLATES)) + 1))[:n_ctx]
            sents = [t.format(w=w) for t in sents]
        out[w] = sents
    return out


def main():
    sen_dic = None
    if SEN_DICT.exists():
        with open(SEN_DICT, "rb") as f:
            sen_dic = pickle.load(f)
        print(f"Loaded Reddit context pickle: {SEN_DICT.name} ({len(sen_dic)} terms)")
    else:
        print("sen_dic_1.pickle missing — using template contexts built from ALL_TESTS")

    # C2 gender is smaller / faster; still real CEAT word lists from data.py
    test = ALL_TESTS[1]
    print(f"Running CEAT on: {test['name']}")
    T1 = _contexts_for_terms(test["t1"], sen_dic)
    T2 = _contexts_for_terms(test["t2"], sen_dic)
    A1 = _contexts_for_terms(test["a1"], sen_dic)
    A2 = _contexts_for_terms(test["a2"], sen_dic)

    model = HuggingFaceModel("bert-base-uncased", task="encoder")
    result = CEAT(sample_size=2, n_trials=20, seed=0).compute(
        model=model,
        T1_contexts=T1,
        T2_contexts=T2,
        A1_contexts=A1,
        A2_contexts=A2,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
