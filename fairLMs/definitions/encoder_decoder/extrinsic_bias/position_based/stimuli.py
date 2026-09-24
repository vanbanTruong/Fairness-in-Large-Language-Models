"""XSum article loader for NPD demos (from e532af2)."""

from __future__ import annotations

from typing import List


def load_xsum_articles(n_max: int = 8) -> List[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("xsum", split="validation")
        arts = []
        for ex in ds:
            doc = (ex.get("document") or "").strip()
            if len(doc.split()) < 40:
                continue
            arts.append(doc[:1200])
            if len(arts) >= n_max:
                break
        print(f"  XSum articles: {len(arts)}")
        return arts
    except Exception as e:
        print(f"  [warn] XSum load failed: {e}")
    arts = [
        "The nurse said the patient was recovering well after surgery at the city hospital. "
        "Doctors credited the rapid response team and careful monitoring overnight.",
        "A software engineer published an open-source tool that helps researchers analyze "
        "large text corpora for demographic bias patterns across languages.",
        "Local teachers organized a community reading night after students requested more "
        "access to science books and after-school tutoring sessions.",
    ]
    print(f"  Fallback articles: {len(arts)}")
    return arts[:n_max]
