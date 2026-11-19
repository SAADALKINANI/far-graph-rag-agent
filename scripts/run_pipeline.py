"""Build the index, train the reranker + scope classifier, run the full evaluation scorecard.

    python scripts/run_pipeline.py
Outputs: artifacts/retriever.pkl, results/scorecard.md, results/metrics.json, results/answers.jsonl, figures.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from farrag.agents import ComplianceAgent, SpendingAgent  # noqa: E402
from farrag.evaluate import (OOS_TRAIN, answer_metrics, load_jsonl, retrieval_metrics,  # noqa: E402
                             router_accuracy)
from farrag.generate import ScopeClassifier  # noqa: E402
from farrag.ingest import build_corpus, load_corpus  # noqa: E402
from farrag.retrieval import Retriever, synthetic_queries  # noqa: E402
from farrag.spending_data import synthetic_awards  # noqa: E402

COMPLIANCE_QS = [
    "We plan to award a $3.2 million sole-source software maintenance services contract, what do we need?",
    "Our office wants to buy $12,000 of laptops, is it allowed with a purchase card?",
    "We need $250,000 of janitorial services, checklist please",
    "Can we sole source a $25 million engineering contract?",
    "We intend to acquire $8 million of commercial cloud services, what procedures apply?",
    "I need a checklist for a $95 million brand name aircraft parts buy",
]


def main() -> None:
    t_start = time.time()
    corpus = ROOT / "data" / "corpus"
    if not (corpus / "chunks.jsonl").exists():
        print(build_corpus(ROOT / "data" / "raw" / "far_html", corpus))
    secs, chunks, G = load_corpus(corpus)
    stats = json.loads((corpus / "corpus_stats.json").read_text())
    golden = load_jsonl(ROOT / "eval" / "golden_qa.jsonl")
    answerable = [g for g in golden if g["intent"] != "out_of_scope"]
    oos = [g for g in golden if g["intent"] == "out_of_scope"]
    gold_sids = {s for g in answerable for s in g["gold"]}

    t0 = time.time()
    retr = Retriever(secs, chunks, G)
    index_s = time.time() - t0
    print(f"index built in {index_s:.1f}s over {len(chunks)} chunks")

    # learned reranker: weak supervision from section titles, gold sections held out
    dev = [{"question": q, "gold": list(g)} for q, g in synthetic_queries(secs, exclude=gold_sids, n=400, seed=99)]
    syn = synthetic_queries(secs, exclude=gold_sids | {d["gold"][0] for d in dev}, n=1200, seed=0)
    rr = retr.train_reranker(syn)
    print("reranker", {k: v for k, v in rr.items() if k != "feature_importance"})

    # in-scope classifier
    scope = ScopeClassifier()
    pos = synthetic_queries(secs, exclude=gold_sids, n=300, seed=1)
    X = [scope.features(retr, q, retr.search(q, k=3)) for q, _ in pos] + \
        [scope.features(retr, q, retr.search(q, k=3)) for q in OOS_TRAIN]
    scope.fit(X, [1] * len(pos) + [0] * len(OOS_TRAIN))

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("far-graph-rag")
    R = {"corpus": stats, "far_source": "GSA/GSA-Acquisition-FAR (HTML edition, repo commit 2026-03-30)",
         "golden_set": {"answerable": len(answerable), "out_of_scope": len(oos)}, "reranker": rr, "retrieval": {}}
    with mlflow.start_run(run_name="eval"):
        R["retrieval_synthetic_dev"] = {}
        for mode in ["bm25", "hybrid", "full"]:
            R["retrieval_synthetic_dev"][mode] = retrieval_metrics(retr, dev, mode)
            print("dev", mode, {k: round(v, 3) for k, v in R["retrieval_synthetic_dev"][mode].items()})
        for mode in ["bm25", "dense", "hybrid", "graph", "full"]:
            m = retrieval_metrics(retr, answerable, mode)
            R["retrieval"][mode] = m
            mlflow.log_metrics({f"{mode}_{k.replace('@', '_at_')}": v for k, v in m.items()})
            print(f"{mode:7s}", {k: round(v, 3) for k, v in m.items()})
        am = answer_metrics(retr, scope, answerable, oos)
        rows = am.pop("_rows")
        R["answers"] = am
        mlflow.log_metrics({f"answer_{k}": v for k, v in am.items()})
        print("answers", {k: round(v, 3) for k, v in am.items()})

        spend_eval = load_jsonl(ROOT / "eval" / "spending_qa.jsonl")
        R["router"] = router_accuracy(golden, [e["q"] for e in spend_eval], COMPLIANCE_QS)
        print("router", R["router"]["accuracy"])

        import sqlite3
        awards = synthetic_awards()
        awards.to_csv(ROOT / "data" / "synthetic_awards.csv.gz", index=False)
        agent = SpendingAgent(awards)
        db = sqlite3.connect(":memory:"); awards.to_sql("awards", db, index=False)
        ok = []
        for e in spend_eval:
            gold = pd.read_sql_query(e["sql"], db).round(2).values.tolist()
            pred = pd.DataFrame(agent.run(e["q"])["rows"]).round(2).values.tolist()
            ok.append(gold == pred)
        blocked = [agent.run(q) for q in ["DROP TABLE awards", "DELETE FROM awards"]]
        guard = agent.db
        try:
            guard.execute("DELETE FROM awards"); write_blocked = False
        except Exception:
            write_blocked = True
        R["nl_to_sql"] = {"execution_accuracy": float(np.mean(ok)), "n": len(ok), "write_statements_blocked": write_blocked}
        print("nl2sql", R["nl_to_sql"])

        ca = ComplianceAgent({s.sid: s for s in secs})
        R["compliance"] = {"thresholds_parsed_from_far_text": {k: v for k, v in ca.t.__dict__.items() if k != "source"},
                           "example": ca.check(COMPLIANCE_QS[0])}
        mlflow.log_metrics({"router_accuracy": R["router"]["accuracy"], "nl2sql_exec_acc": R["nl_to_sql"]["execution_accuracy"]})

    with open(ROOT / "artifacts" / "retriever.pkl", "wb") as f:
        pickle.dump({"retriever": retr, "scope": scope}, f)
    (ROOT / "results" / "answers.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    R["total_seconds"] = time.time() - t_start
    (ROOT / "results" / "metrics.json").write_text(json.dumps(R, indent=2, default=float))
    scorecard(R)
    plot(R)


def scorecard(R):
    L = ["# Evaluation scorecard", "",
         f"Corpus: {R['corpus']['sections']:,} FAR units, {R['corpus']['chunks']:,} chunks, "
         f"{R['corpus']['citation_edges']:,} citation edges, {R['corpus']['words']:,} words. Source: {R['far_source']}.", "",
         f"Golden set: {R['golden_set']['answerable']} hand-written questions with gold FAR sections "
         f"(every gold section auto-verified to contain the expected fact) + {R['golden_set']['out_of_scope']} out-of-scope questions.", "",
         "## Retrieval (section-level)", "", "| Mode | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 latency |", "|---|---|---|---|---|---|---|"]
    names = {"bm25": "BM25", "dense": "Dense (LSA)", "hybrid": "Hybrid RRF", "graph": "Hybrid + citation graph",
             "full": "**Hybrid + graph + LTR reranker**"}
    for k, m in R["retrieval"].items():
        L.append(f"| {names[k]} | {m['recall@1']:.1%} | {m['recall@5']:.1%} | {m['recall@10']:.1%} | {m['mrr@10']:.3f} | "
                 f"{m['ndcg@10']:.3f} | {m['latency_p50_ms']:.0f} ms |")
    L += ["", "Held-out *synthetic* dev set (400 auto-generated questions on sections never used for training):", "",
          "| Mode | Recall@1 | Recall@5 | MRR@10 |", "|---|---|---|---|"]
    for k, m in R["retrieval_synthetic_dev"].items():
        L.append(f"| {names[k]} | {m['recall@1']:.1%} | {m['recall@5']:.1%} | {m['mrr@10']:.3f} |")
    a = R["answers"]
    L += ["", "## Answers (offline extractive generator - zero LLM calls)", "",
          "| Metric | Value |", "|---|---|",
          f"| Key-fact recall (answer contains the gold fact) | {a['key_fact_recall']:.1%} |",
          f"| Answers with citations | {a['answers_with_citations']:.1%} |",
          f"| Citation validity (cited section exists) | {a['citation_validity']:.1%} |",
          f"| Cites a gold section | {a['cites_a_gold_section']:.1%} |",
          f"| Out-of-scope questions correctly refused | {a['out_of_scope_abstention_rate']:.1%} |",
          f"| False refusals on answerable questions | {a['false_abstention_rate']:.1%} |",
          f"| End-to-end latency p50 / p95 | {a['latency_p50_ms']:.0f} / {a['latency_p95_ms']:.0f} ms |",
          "", "## Agents", "",
          f"* Router accuracy: **{R['router']['accuracy']:.1%}** over {R['router']['n']} requests (regulation / compliance / spending)",
          f"* NL→SQL execution accuracy: **{R['nl_to_sql']['execution_accuracy']:.1%}** ({R['nl_to_sql']['n']} questions); write statements blocked: {R['nl_to_sql']['write_statements_blocked']}",
          "* Compliance thresholds parsed live from FAR text: " + ", ".join(
              f"{k} = ${v:,.0f}" for k, v in R["compliance"]["thresholds_parsed_from_far_text"].items())]
    (ROOT / "results" / "scorecard.md").write_text("\n".join(L) + "\n")


def plot(R):
    modes = list(R["retrieval"])
    lab = ["BM25", "Dense\n(LSA)", "Hybrid\nRRF", "+ citation\ngraph", "+ LTR\nreranker"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(modes))
    for off, k, c in [(-0.2, "recall@1", "#9bb7e0"), (0, "recall@5", "#5b8bd6"), (0.2, "mrr@10", "#1f4e9c")]:
        ax.bar(x + off, [R["retrieval"][m][k] for m in modes], 0.2, label=k, color=c)
    ax.set_xticks(x, lab); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Retrieval ablation on hand-verified FAR questions")
    fig.tight_layout(); fig.savefig(ROOT / "results" / "figures" / "retrieval_ablation.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    (ROOT / "artifacts").mkdir(exist_ok=True); (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    main()
