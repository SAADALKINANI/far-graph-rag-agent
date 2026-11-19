"""Build (or load) the retriever + scope classifier used by the app.

The fitted index is ~190 MB, too big for git, so a fresh clone rebuilds it from the committed parsed corpus
(``data/corpus``) on first launch (~2 minutes on a laptop) and caches it in ``artifacts/retriever.pkl``.
The procedure is identical to ``scripts/run_pipeline.py``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from .evaluate import OOS_TRAIN
from .generate import ScopeClassifier
from .ingest import build_corpus, load_corpus
from .retrieval import Retriever, synthetic_queries


def load_or_build(root: str | Path):
    root = Path(root)
    pkl = root / "artifacts" / "retriever.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            b = pickle.load(f)
        return b["retriever"], b["scope"]
    corpus = root / "data" / "corpus"
    if not (corpus / "chunks.jsonl").exists():
        raw = root / "data" / "raw" / "far_html"
        if not raw.exists():
            raise FileNotFoundError("FAR corpus not found. Run: bash scripts/download_data.sh")
        build_corpus(raw, corpus)
    secs, chunks, G = load_corpus(corpus)
    golden = [json.loads(line) for line in (root / "eval" / "golden_qa.jsonl").read_text().splitlines() if line.strip()]
    gold_sids = {s for g in golden for s in g["gold"]}
    dev = synthetic_queries(secs, exclude=gold_sids, n=400, seed=99)
    retr = Retriever(secs, chunks, G)
    retr.train_reranker(synthetic_queries(secs, exclude=gold_sids | {next(iter(g)) for _, g in dev}, n=1200, seed=0))
    scope = ScopeClassifier()
    pos = synthetic_queries(secs, exclude=gold_sids, n=300, seed=1)
    X = [scope.features(retr, q, retr.search(q, k=3)) for q, _ in pos] + \
        [scope.features(retr, q, retr.search(q, k=3)) for q in OOS_TRAIN]
    scope.fit(X, [1] * len(pos) + [0] * len(OOS_TRAIN))
    pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl, "wb") as f:
        pickle.dump({"retriever": retr, "scope": scope}, f)
    return retr, scope
