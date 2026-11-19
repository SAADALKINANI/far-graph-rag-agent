"""Hybrid + graph-aware retrieval with a learning-to-rank reranker.

Pipeline for a query q:
  1. Sparse:  BM25 over "header + text" (catches exact regulatory language and numbers).
  2. Dense:   semantic embeddings (pluggable: sentence-transformers if available, otherwise a
              self-contained LSA model - TF-IDF bigrams + truncated SVD - so the system runs in an
              air-gapped enclave with zero model downloads).
  3. Fusion:  Reciprocal Rank Fusion (Cormack et al., 2009) of the two lists.
  4. Graph:   personalised PageRank over the FAR citation graph, seeded with the fused results,
              pulls in sections that the top hits *cite* or are cited by (Graph-RAG). Explicit
              section numbers in the query ("15.404-4", "subpart 19.8") are resolved directly.
  5. Rerank:  LightGBM LambdaRank model over ~15 query-chunk features, trained with weak supervision
              (synthetic queries generated from section titles; evaluation sections held out).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import lightgbm as lgb
import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .ingest import Chunk, Section

STOP = set("""a an the of to in for on by with and or is are be was were this that these those it its as at from
any all may shall must will which who whom what when where how does do did not no under than such other each
can could would should if into about over per their there then they them his her he she you your we our""".split())
SID_RE = re.compile(r"\b(\d{1,2}\.\d{3,4}(?:-\d{1,3})?)\b")
SUBPART_Q = re.compile(r"\bsubpart\s+(\d{1,2}\.\d{1,2})\b", re.I)
PART_Q = re.compile(r"\bpart\s+(\d{1,2})\b(?!\.)", re.I)


def stem(w: str) -> str:
    for suf in ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied", "es", "ed", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"\d{1,2}\.\d{3,4}(?:-\d{1,3})?|[a-z0-9]+", text.lower())
    return [t if "." in t else stem(t) for t in toks if t not in STOP and len(t) > 1]


class LSAEmbedder:
    """Offline semantic embeddings: TF-IDF (word uni+bigrams) -> 256-d truncated SVD."""

    def __init__(self, dim: int = 256, seed: int = 0):
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2), min_df=2,
                                   max_df=0.5, sublinear_tf=True, token_pattern=None)
        self.svd = TruncatedSVD(dim, random_state=seed)

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        self.svd.fit(self.vec.fit_transform(texts))
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return normalize(self.svd.transform(self.vec.transform(texts))).astype(np.float32)


class STEmbedder:
    """Drop-in neural embedder (e.g. BAAI/bge-small-en-v1.5) when model weights are available."""

    def __init__(self, name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.m = SentenceTransformer(name)

    def fit(self, texts):
        return self

    def encode(self, texts):
        return self.m.encode(texts, normalize_embeddings=True, batch_size=64)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    features: dict


def rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    out: dict[int, float] = {}
    for lst in rank_lists:
        for r, i in enumerate(lst):
            out[i] = out.get(i, 0.0) + 1.0 / (k + r + 1)
    return out


class Retriever:
    FEATURES = ["bm25", "bm25_raw", "bm25_rank", "dense", "dense_rank", "rrf", "ppr", "title_overlap", "header_overlap",
                "sid_match", "part_match", "first_chunk", "depth", "len_log", "is_definitions", "graph_only",
                "is_clause", "section_support", "section_bm25_max", "def_term_match"]

    def __init__(self, sections: list[Section], chunks: list[Chunk], graph: nx.DiGraph, embedder=None):
        self.sections = {s.sid: s for s in sections}
        self.chunks = chunks
        self.G = graph
        self.by_sid: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            self.by_sid.setdefault(c.sid, []).append(i)
        docs = [f"{c.header}\n{c.text}" for c in chunks]
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])
        self.embedder = (embedder or LSAEmbedder()).fit(docs)
        self.E = self.embedder.encode(docs)
        self.title_tok = [set(tokenize(c.title)) for c in chunks]
        self.header_tok = [set(tokenize(c.header)) for c in chunks]
        self.ranker: lgb.LGBMRanker | None = None
        self.ug = graph.to_undirected()
        self.global_pr = nx.pagerank(self.ug, alpha=0.75) if len(self.ug) else {}   # hub prior

    # ------------------------------------------------------------------ candidate generation
    def candidates(self, query: str, k_each: int = 40, use_graph: bool = True) -> list[Hit]:
        qt = tokenize(query)
        bm = self.bm25.get_scores(qt)
        bm_rank = np.argsort(-bm)[:k_each]
        qe = self.embedder.encode([query])[0]
        dn = self.E @ qe
        dn_rank = np.argsort(-dn)[:k_each]
        fused = rrf([bm_rank.tolist(), dn_rank.tolist()])

        # explicit citations in the query
        explicit = set(SID_RE.findall(query)) | {f"Subpart {m}" for m in SUBPART_Q.findall(query)}
        parts = {int(p) for p in PART_Q.findall(query)}
        for sid in explicit:
            for i in self.by_sid.get(sid, [])[:3]:
                fused[i] = fused.get(i, 0) + 0.05
            for child in self.G.successors(sid) if sid in self.G else []:
                for i in self.by_sid.get(child, [])[:1]:
                    fused[i] = fused.get(i, 0) + 0.01

        ppr = {}
        graph_only = set()
        if use_graph:
            seeds: dict[str, float] = {}
            for i, s in sorted(fused.items(), key=lambda kv: -kv[1])[:15]:
                seeds[self.chunks[i].sid] = seeds.get(self.chunks[i].sid, 0) + s
            local = nx.ego_graph(self.ug, list(seeds)[0], radius=2) if seeds else self.ug
            nodes = set(local) | set(seeds)
            for s in list(seeds):
                nodes |= set(self.ug.neighbors(s)) if s in self.ug else set()
            sub = self.ug.subgraph(nodes)
            pers = {n: seeds.get(n, 0.0) for n in sub}
            if sum(pers.values()) > 0:
                raw = nx.pagerank(sub, alpha=0.75, personalization=pers, max_iter=50, tol=1e-4)
                # relative lift over the node's global PageRank: removes hub bias (e.g. 2.101 Definitions,
                # which almost every section cites, would otherwise be pulled into every result list)
                gmean = 1.0 / max(1, len(self.ug))
                ppr = {n: float(np.log1p(v / (self.global_pr.get(n, gmean) * len(sub) / max(1, len(self.ug)) + 1e-12)))
                       for n, v in raw.items()}
                for sid, v in sorted(ppr.items(), key=lambda kv: -kv[1])[:6]:
                    for i in self.by_sid.get(sid, [])[:1]:
                        if i not in fused:
                            graph_only.add(i)
                            fused[i] = 0.0

        qset = set(qt)
        qlow = " " + " ".join(re.findall(r"[a-z0-9]+", query.lower())) + " "
        sec_support: dict[str, int] = {}
        sec_bm: dict[str, float] = {}
        for i in fused:
            sid = self.chunks[i].sid
            sec_support[sid] = sec_support.get(sid, 0) + 1
            sec_bm[sid] = max(sec_bm.get(sid, 0.0), float(bm[i]))
        bm_pos = {int(i): r for r, i in enumerate(np.argsort(-bm)[:500])}
        dn_pos = {int(i): r for r, i in enumerate(np.argsort(-dn)[:500])}
        bmax = bm.max() + 1e-9
        hits = []
        for i, f in fused.items():
            c = self.chunks[i]
            sec = self.sections.get(c.sid)
            feats = {
                "bm25": bm[i] / bmax, "bm25_rank": np.log1p(bm_pos.get(i, 500)), "dense": float(dn[i]),
                "dense_rank": np.log1p(dn_pos.get(i, 500)), "rrf": f, "ppr": ppr.get(c.sid, 0.0),
                "title_overlap": len(qset & self.title_tok[i]) / (len(qset) + 1e-9),
                "header_overlap": len(qset & self.header_tok[i]) / (len(qset) + 1e-9),
                "sid_match": float(c.sid in explicit or any(c.sid.startswith(e + "-") for e in explicit)),
                "part_match": float(sec is not None and sec.part in parts),
                "first_chunk": float(c.cid.endswith("#0")), "depth": float(c.sid.count(".") + c.sid.count("-")),
                "len_log": np.log1p(len(c.text.split())), "is_definitions": float(c.sid in ("2.101", "2.201")),
                "graph_only": float(i in graph_only),
                "bm25_raw": float(np.log1p(bm[i])),
                "is_clause": float(c.sid.startswith(("52.", "53."))),
                "section_support": float(sec_support.get(c.sid, 0)),
                "section_bm25_max": sec_bm.get(c.sid, 0.0) / bmax,
                "def_term_match": float(c.sid == "2.101" and len(c.title) > 3 and
                                        (" " + " ".join(re.findall(r"[a-z0-9]+", c.title.lower())) + " ") in qlow),
            }
            hits.append(Hit(c, f, feats))
        return sorted(hits, key=lambda h: -h.score)

    # ------------------------------------------------------------------ public API
    def search(self, query: str, k: int = 8, mode: str = "full") -> list[Hit]:
        """mode: bm25 | dense | hybrid | graph | full (graph + learned reranker)."""
        if mode in ("bm25", "dense"):
            qt = tokenize(query)
            s = self.bm25.get_scores(qt) if mode == "bm25" else self.E @ self.embedder.encode([query])[0]
            return [Hit(self.chunks[i], float(s[i]), {}) for i in np.argsort(-s)[:k]]
        hits = self.candidates(query, use_graph=mode in ("graph", "full"))
        if mode == "full" and self.ranker is not None:
            X = np.array([[h.features[f] for f in self.FEATURES] for h in hits])
            for h, s in zip(hits, self.ranker.predict(X)):
                h.score = float(s)
            hits.sort(key=lambda h: -h.score)
        elif mode == "graph":
            for h in hits:
                h.score = h.features["rrf"] + 0.002 * h.features["ppr"] - 0.006 * h.features["graph_only"]
            hits.sort(key=lambda h: -h.score)
        return hits[:k]

    # ------------------------------------------------------------------ reranker training
    def train_reranker(self, queries: list[tuple[str, set[str]]], seed: int = 0) -> dict:
        X, y, groups = [], [], []
        for q, gold in queries:
            hits = self.candidates(q)
            if not any(h.chunk.sid in gold for h in hits):
                continue
            for h in hits:
                X.append([h.features[f] for f in self.FEATURES])
                y.append(2 if (h.chunk.sid in gold and h.chunk.cid.endswith("#0")) else int(h.chunk.sid in gold))
            groups.append(len(hits))
        self.ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15,
                                     min_child_samples=20, subsample=0.8, subsample_freq=1, random_state=seed, verbose=-1)
        self.ranker.fit(np.array(X), np.array(y), group=groups)
        imp = dict(zip(self.FEATURES, self.ranker.booster_.feature_importance("gain")))
        tot = sum(imp.values()) or 1
        return {"train_queries": len(groups), "pairs": len(y),
                "feature_importance": {k: round(v / tot, 4) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}}


def synthetic_queries(sections: list[Section], exclude: set[str], n: int = 1500, seed: int = 0) -> list[tuple[str, set[str]]]:
    """Weak supervision: turn section titles (+ a sentence fragment) into natural-ish questions."""
    rng = np.random.default_rng(seed)
    tmpl = ["What does the FAR say about {t}?", "{t}", "requirements for {t}", "When is {t} required?",
            "Explain the rules on {t}", "guidance on {t} for contracting officers", "{t} policy"]
    pool = [s for s in sections if s.level == "section" and s.text and s.sid not in exclude and len(s.title) > 8
            and not s.sid.startswith("53.") and (not s.sid.startswith("52.") or rng.random() < 0.15)]
    qwords = ["What", "When", "Who", "How", "Does", "Must", "Can"]
    out = []
    for s in rng.choice(pool, size=min(n, len(pool)), replace=False):
        if rng.random() < 0.5:                                   # content question from a body sentence
            sents = [x for x in re.split(r"(?<=[.;])\s+", s.text[:3000]) if len(x.split()) > 8]
            if sents:
                words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", rng.choice(sents)) if w.lower() not in STOP]
                if len(words) >= 4:
                    k = int(rng.integers(4, min(9, len(words)) + 1))
                    start = int(rng.integers(0, len(words) - k + 1))
                    out.append((f"{rng.choice(qwords)} {' '.join(words[start:start + k]).lower()}?", {s.sid}))
                    continue
        t = re.sub(r"[.]$", "", s.title).lower()
        words = t.split()
        if len(words) > 4 and rng.random() < 0.4:              # token dropout -> paraphrase-ish
            words = [w for w in words if rng.random() > 0.25]
        q = rng.choice(tmpl).format(t=" ".join(words))
        if rng.random() < 0.3:                                    # add a content word from the body
            body = [w for w in tokenize(s.text[:400]) if len(w) > 5]
            if body:
                q += " " + rng.choice(body)
        out.append((q, {s.sid}))
    return out
