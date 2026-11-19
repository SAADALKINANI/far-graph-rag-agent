"""Grounded answer generation with citation guardrails and calibrated abstention.

Backends
  * ``ExtractiveGenerator`` (default, offline): selects the most query-relevant sentences from
    the retrieved FAR text and cites each one. Zero hallucination by construction - useful as
    an air-gapped fallback and as the faithfulness ceiling for the LLM path.
  * ``LLMGenerator``: Anthropic / OpenAI / any OpenAI-compatible endpoint (Ollama, vLLM), with
    a strict citation prompt. Every answer is post-validated: cited sections must exist in the
    FAR *and* be in the retrieved context; uncited or invalid answers are replaced by the
    extractive answer.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from .retrieval import Hit, Retriever, tokenize

CITE_RE = re.compile(r"\[FAR ([^\]]+)\]")
PROMPT = """You are a federal acquisition policy assistant. Answer ONLY from the FAR excerpts below.
Cite every claim with the section in square brackets, e.g. [FAR 15.404-4]. If the excerpts do not
answer the question, reply exactly: "I could not find this in the FAR excerpts provided."
Keep it under 150 words. This is not legal advice.

QUESTION: {q}

EXCERPTS:
{ctx}
"""


@dataclass
class Answer:
    text: str
    citations: list[str]
    abstained: bool
    confidence: float
    backend: str
    hits: list[Hit]
    guardrail_notes: list[str]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;:])\s+(?=[A-Z(])|\n", text)
    return [p.strip() for p in parts if len(p.split()) >= 5]


class ScopeClassifier:
    """Calibrated in-scope detector: is the question answerable from the FAR at all?"""
    FEATS = ["max_bm25_raw", "max_dense", "top_score", "top_title_overlap", "top3_dense_gap"]

    def __init__(self):
        self.clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)

    def features(self, retr: Retriever, q: str, hits: list[Hit]) -> list[float]:
        raw = retr.bm25.get_scores(tokenize(q))
        dense = retr.E @ retr.embedder.encode([q])[0]
        ds = np.sort(dense)[::-1]
        return [float(np.log1p(raw.max())), float(ds[0]), float(hits[0].score if hits else 0),
                float(hits[0].features.get("title_overlap", 0) if hits else 0), float(ds[0] - ds[2])]

    def fit(self, X, y):
        self.clf.fit(np.array(X), np.array(y))
        return self

    def prob(self, x) -> float:
        return float(self.clf.predict_proba(np.array([x]))[0, 1])


class ExtractiveGenerator:
    name = "extractive"

    def __call__(self, q: str, hits: list[Hit], max_sent: int = 4) -> str:
        qt = set(tokenize(q))
        # definitional questions ("What is the X?", "Define X") -> prefer the sentence "X means ..."
        m = re.match(r"^\s*(?:what (?:is|are)|define|definition of)\s+(?:the |a |an )?(.+?)\s*\??$", q, re.I)
        term = m.group(1).lower() if m else None
        cands = []
        for r, h in enumerate(hits[:5]):
            for j, s in enumerate(split_sentences(h.chunk.text)):
                st = set(tokenize(s))
                if not st:
                    continue
                overlap = len(qt & st) / (len(qt) ** 0.5 * len(st) ** 0.25 + 1e-9)
                if term and s.lower().startswith(term) and " means" in s.lower()[:len(term) + 12]:
                    overlap += 1.0
                nums = 0.15 * bool(re.search(r"\$[\d,.]+|\d+ (days|years)", s)) * bool(
                    qt & {"much", "threshold", "amount", "long", "many", "day", "year", "maximum", "period"})
                cands.append((overlap + nums - 0.08 * r - 0.01 * j, r, j, s, h.chunk.sid))
        cands.sort(key=lambda c: -c[0])
        top = cands[:max_sent]
        lead = [c for c in top if c[0] >= 1.0][:1]                  # a matched definition goes first
        chosen = lead + sorted([c for c in top if c not in lead], key=lambda c: (c[1], c[2]))
        return " ".join(f"{s} [FAR {sid}]" for *_, s, sid in chosen)


class LLMGenerator:
    def __init__(self, backend: str = "anthropic", model: str | None = None):
        self.name, self.model = backend, model

    def __call__(self, q: str, hits: list[Hit]) -> str:
        ctx = "\n\n".join(f"[FAR {h.chunk.sid}] {h.chunk.header}\n{h.chunk.text[:1500]}" for h in hits[:6])
        prompt = PROMPT.format(q=q, ctx=ctx)
        if self.name == "anthropic":
            import anthropic
            r = anthropic.Anthropic().messages.create(model=self.model or "claude-sonnet-4-5", max_tokens=500,
                                                      messages=[{"role": "user", "content": prompt}])
            return r.content[0].text
        from openai import OpenAI
        r = OpenAI(base_url=os.getenv("OPENAI_BASE_URL")).chat.completions.create(
            model=self.model or "gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0)
        return r.choices[0].message.content


def default_generator():
    if os.getenv("ANTHROPIC_API_KEY"):
        return LLMGenerator("anthropic")
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
        return LLMGenerator("openai")
    return ExtractiveGenerator()


def answer(q: str, retr: Retriever, scope: ScopeClassifier | None = None, gen=None, k: int = 6,
           abstain_below: float = 0.5) -> Answer:
    gen = gen or default_generator()
    hits = retr.search(q, k=k)
    notes = []
    conf = scope.prob(scope.features(retr, q, hits)) if scope else 1.0
    if conf < abstain_below:
        return Answer("I could not find this in the FAR. It may be outside the scope of the Federal Acquisition "
                      "Regulation (e.g. agency supplements, GSA travel rules or non-procurement topics).",
                      [], True, conf, getattr(gen, "name", "?"), hits, ["abstained: low in-scope confidence"])
    try:
        text = gen(q, hits)
    except Exception as e:  # network/auth failure -> offline fallback
        notes.append(f"LLM backend failed ({type(e).__name__}); used extractive fallback")
        gen = ExtractiveGenerator(); text = gen(q, hits)
    cites = [c.strip() for c in CITE_RE.findall(text)]
    allowed = {h.chunk.sid for h in hits}
    bad = [c for c in cites if c not in retr.sections or c not in allowed]
    if (not cites or bad) and not isinstance(gen, ExtractiveGenerator):
        notes.append(f"citation guardrail triggered (invalid={bad}, cited={len(cites)}); used extractive answer")
        text = ExtractiveGenerator()(q, hits)
        cites = [c.strip() for c in CITE_RE.findall(text)]
    return Answer(text, sorted(set(cites)), False, conf, getattr(gen, "name", "?"), hits, notes)
