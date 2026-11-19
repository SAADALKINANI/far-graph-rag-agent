# Evaluation scorecard

Corpus: 3,868 FAR units, 5,690 chunks, 6,215 citation edges, 939,476 words. Source: GSA/GSA-Acquisition-FAR (HTML edition, repo commit 2026-03-30).

Golden set: 70 hand-written questions with gold FAR sections (every gold section auto-verified to contain the expected fact) + 5 out-of-scope questions.

## Retrieval (section-level)

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|
| BM25 | 55.7% | 88.6% | 91.4% | 0.701 | 0.754 | 7 ms |
| Dense (LSA) | 41.4% | 77.1% | 84.3% | 0.556 | 0.626 | 42 ms |
| Hybrid RRF | 55.7% | 84.3% | 91.4% | 0.686 | 0.742 | 53 ms |
| Hybrid + citation graph | 52.9% | 84.3% | 91.4% | 0.664 | 0.725 | 61 ms |
| **Hybrid + graph + LTR reranker** | 54.3% | 87.1% | 91.4% | 0.677 | 0.735 | 69 ms |

Held-out *synthetic* dev set (400 auto-generated questions on sections never used for training):

| Mode | Recall@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| BM25 | 56.0% | 76.5% | 0.649 |
| Hybrid RRF | 38.5% | 64.8% | 0.496 |
| **Hybrid + graph + LTR reranker** | 64.5% | 80.8% | 0.716 |

## Answers (offline extractive generator - zero LLM calls)

| Metric | Value |
|---|---|
| Key-fact recall (answer contains the gold fact) | 80.0% |
| Answers with citations | 94.3% |
| Citation validity (cited section exists) | 100.0% |
| Cites a gold section | 84.8% |
| Out-of-scope questions correctly refused | 80.0% |
| False refusals on answerable questions | 5.7% |
| End-to-end latency p50 / p95 | 130 / 163 ms |

## Agents

* Router accuracy: **100.0%** over 92 requests (regulation / compliance / spending)
* NL→SQL execution accuracy: **100.0%** (16 questions); write statements blocked: True
* Compliance thresholds parsed live from FAR text: micro_purchase = $15,000, simplified = $350,000, commercial_simplified_cap = $9,000,000, certified_cost_pricing = $2,500,000, ja_co_max = $900,000, ja_competition_advocate_max = $20,000,000, ja_hca_max = $90,000,000
