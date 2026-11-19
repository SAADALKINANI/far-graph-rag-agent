"""Evaluation harness: retrieval quality, answer quality, abstention, routing and NL->SQL."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from .agents import route
from .generate import CITE_RE, ExtractiveGenerator, answer

# Generic out-of-scope questions used ONLY to train the scope classifier (disjoint from eval set).
OOS_TRAIN = [
    "What is the weather in Chicago tomorrow?", "How many calories are in an apple?", "Who wrote Pride and Prejudice?",
    "Best way to learn Python programming", "What is the tallest mountain in Africa?", "How do I change a car tire?",
    "Explain quantum entanglement simply", "What time zone is Tokyo in?", "Translate hello into Spanish",
    "What are the symptoms of the flu?", "How do vaccines work?", "Who is the CEO of Microsoft?",
    "What is the federal income tax rate for 2025?", "How do I apply for a passport?", "What is a Roth IRA?",
    "Plan a 3 day trip to Paris", "What is the speed of light?", "How to train a puppy",
    "Recipe for chocolate chip cookies", "What year did World War II end?", "How do I reset my router?",
    "What is the GDP of Japan?", "Who painted the Mona Lisa?", "How many players on a soccer team?",
    "What is photosynthesis?", "How do I file for unemployment benefits?", "What does HIPAA protect?",
    "What is the minimum wage in California?", "How do I renew my driver's license?", "Explain machine learning",
    "What is the federal per diem rate for Denver lodging?", "When is the next leap year?",
    "What's a good name for a cat?", "How do solar panels work?", "What is inflation?", "Who invented the telephone?",
    "How many bones are in the human body?", "What are OPM leave rules for federal employees?",
    "How do I get a security clearance?", "What is the Hatch Act?",
]


def section_hit(sid: str, gold: list[str]) -> bool:
    """A retrieved section is relevant if it is a gold section, or lies inside a gold *subpart*
    (FAR numbering: subpart 19.8 contains sections 19.800-19.899, subpart 19.13 -> 19.1300-19.1399)."""
    for g in gold:
        if sid == g:
            return True
        if g.startswith("Subpart "):
            part, sub = g.split(" ")[1].split(".")
            if re.match(rf"^{part}\.{sub}\d{{2}}(-|$)", sid):
                return True
    return False


def retrieval_metrics(retr, golden: list[dict], mode: str, k: int = 10) -> dict:
    r1 = r5 = r10 = mrr = ndcg = 0.0
    lat = []
    for g in golden:
        t0 = time.time()
        hits = retr.search(g["question"], k=k, mode=mode)
        lat.append(time.time() - t0)
        sids = []
        for h in hits:                          # de-duplicate to section level, keep order
            if h.chunk.sid not in sids:
                sids.append(h.chunk.sid)
        rel = [section_hit(s, g["gold"]) for s in sids[:k]]
        first = next((i for i, r in enumerate(rel) if r), None)
        r1 += first is not None and first < 1
        r5 += first is not None and first < 5
        r10 += first is not None
        mrr += 1 / (first + 1) if first is not None else 0
        ndcg += 1 / np.log2(first + 2) if first is not None else 0
    n = len(golden)
    return {"recall@1": r1 / n, "recall@5": r5 / n, "recall@10": r10 / n, "mrr@10": mrr / n, "ndcg@10": ndcg / n,
            "latency_p50_ms": float(np.percentile(lat, 50) * 1000), "latency_p95_ms": float(np.percentile(lat, 95) * 1000)}


def answer_metrics(retr, scope, golden: list[dict], oos: list[dict], gen=None) -> dict:
    gen = gen or ExtractiveGenerator()
    fact_hits, cite_valid, cite_gold, false_abstain, n_cited, lat, rows = 0, 0, 0, 0, 0, [], []
    for g in golden:
        t0 = time.time()
        a = answer(g["question"], retr, scope, gen)
        lat.append(time.time() - t0)
        cited = [c.strip() for c in CITE_RE.findall(a.text)]
        false_abstain += a.abstained
        fh = all(f.lower() in a.text.lower() for f in g["facts"]) if not a.abstained else False
        fact_hits += fh
        if cited:
            n_cited += 1
            cite_valid += all(c in retr.sections for c in cited)
            cite_gold += any(section_hit(c, g["gold"]) for c in cited)
        rows.append({"id": g["id"], "question": g["question"], "answer": a.text, "facts_found": fh,
                     "citations": a.citations, "gold": g["gold"], "abstained": a.abstained, "confidence": round(a.confidence, 3)})
    oos_abstain = sum(answer(o["question"], retr, scope, gen).abstained for o in oos)
    n = len(golden)
    return {"key_fact_recall": fact_hits / n, "answers_with_citations": n_cited / n,
            "citation_validity": cite_valid / max(1, n_cited), "cites_a_gold_section": cite_gold / max(1, n_cited),
            "false_abstention_rate": false_abstain / n, "out_of_scope_abstention_rate": oos_abstain / max(1, len(oos)),
            "latency_p50_ms": float(np.percentile(lat, 50) * 1000), "latency_p95_ms": float(np.percentile(lat, 95) * 1000),
            "_rows": rows}


def router_accuracy(golden: list[dict], spending_qs: list[str], compliance_qs: list[str]) -> dict:
    cases = [(g["question"], "regulation") for g in golden if g["intent"] == "regulation"]
    cases += [(q, "spending") for q in spending_qs] + [(q, "compliance") for q in compliance_qs]
    correct = [route(q) == y for q, y in cases]
    return {"accuracy": float(np.mean(correct)), "n": len(cases),
            "errors": [{"q": q, "expected": y, "got": route(q)} for (q, y), c in zip(cases, correct) if not c]}


def load_jsonl(p: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
