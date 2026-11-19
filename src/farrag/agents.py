"""Multi-agent layer: a router dispatches each request to a specialist.

  RegulationAgent  - Graph-RAG question answering over the FAR (retrieval + grounded generation).
  ComplianceAgent  - checks a *described acquisition* against FAR thresholds. Every threshold is
                     parsed from the live regulation text at start-up (never hard-coded), so when
                     the FAR is amended (e.g. the October 2025 inflation adjustments) the agent
                     updates by re-ingesting - no code change.
  SpendingAgent    - natural-language -> SQL over a USAspending-schema contract-awards table,
                     with a read-only SQL guard (SQLite authorizer: SELECT on whitelisted tables).
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from .generate import answer
from .ingest import Section

MONEY = r"\$\s?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|[mkb])?\b"


def to_dollars(num: str, unit: str | None) -> float:
    v = float(num.replace(",", ""))
    mult = {"million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9, "thousand": 1e3, "k": 1e3}
    return v * mult.get((unit or "").lower(), 1)


# ------------------------------------------------------------------------------------------------
# Router
# ------------------------------------------------------------------------------------------------
SPEND_RE = re.compile(r"\b(spend|spent|spending|obligat\w*|awarded to|top (\d+ )?(vendors|recipients|contractors)|"
                      r"how many (contracts|awards)|total (dollars|value)|usaspending|which agenc\w+ (spent|awarded)|"
                      r"share of .*(sole.source|set.aside)|by fiscal year|by agency)\b", re.I)
COMPLY_RE = re.compile(r"\b(we (plan|want|need|intend)|our (agency|office|program|requirement)|i (plan|want|need)|"
                       r"checklist|what do (we|i) need|is (this|it) allowed|can we|planning to (buy|award|acquire|procure))\b", re.I)


def route(q: str) -> str:
    if SPEND_RE.search(q):
        return "spending"
    if COMPLY_RE.search(q) and re.search(MONEY, q, re.I):
        return "compliance"
    return "regulation"


# ------------------------------------------------------------------------------------------------
# Compliance agent
# ------------------------------------------------------------------------------------------------
@dataclass
class Thresholds:
    micro_purchase: float
    simplified: float
    commercial_simplified_cap: float
    certified_cost_pricing: float
    ja_co_max: float
    ja_competition_advocate_max: float
    ja_hca_max: float
    source: dict = field(default_factory=dict)


def _grab(text: str, pattern: str) -> tuple[float, str]:
    m = re.search(pattern, text, re.S)
    if not m:
        raise ValueError(f"pattern not found: {pattern}")
    return to_dollars(m.group(1), m.group(2) if m.lastindex and m.lastindex >= 2 else None), m.group(0)[:160]


def parse_thresholds(sections: dict[str, Section]) -> Thresholds:
    d = sections
    mpt, s1 = _grab(d["2.101"].text, r"Micro-purchase threshold means \$([\d,]+)()")
    sat, s2 = _grab(d["2.101"].text, r"Simplified acquisition threshold means \$([\d,]+)()")
    com, s3 = _grab(d["13.500"].text, r"not exceeding \$([\d.]+) (million)")
    tina, s4 = _grab(d["15.403-4"].text, r"and \$([\d.]+) (million) for prime contracts awarded on or after")
    co, s5 = _grab(d["6.304"].text, r"proposed contract not exceeding \$([\d,]+)()")
    adv, s6 = _grab(d["6.304"].text, r"over \$[\d,]+ but not exceeding \$([\d.]+) (million)")
    hca, s7 = _grab(d["6.304"].text, r"over \$[\d.]+ million, but not exceeding \$([\d.]+) (million)")
    return Thresholds(mpt, sat, com, tina, co, adv, hca, source={
        "micro_purchase": ("2.101", s1), "simplified": ("2.101", s2), "commercial_simplified_cap": ("13.500", s3),
        "certified_cost_pricing": ("15.403-4", s4), "ja_co_max": ("6.304", s5),
        "ja_competition_advocate_max": ("6.304", s6), "ja_hca_max": ("6.304", s7)})


class ComplianceAgent:
    def __init__(self, sections: dict[str, Section]):
        self.t = parse_thresholds(sections)

    def check(self, q: str) -> dict:
        m = re.search(MONEY, q, re.I)
        if not m:
            return {"error": "Please include the estimated dollar value (e.g. $2.4 million)."}
        value = to_dollars(m.group(1), m.group(2))
        ql = q.lower()
        commercial = bool(re.search(r"\bcommercial\b", ql)) and "non-commercial" not in ql and "noncommercial" not in ql
        sole = bool(re.search(r"sole.source|single source|only one (source|vendor)|brand.name|no competition", ql))
        services = bool(re.search(r"\bservices?\b|support|maintenance|consult", ql))
        t, items = self.t, []

        def add(topic, finding, sid):
            items.append({"topic": topic, "finding": finding, "citation": f"FAR {sid}"})

        if value <= t.micro_purchase:
            add("Procedure", f"${value:,.0f} is at or below the micro-purchase threshold (${t.micro_purchase:,.0f}); "
                "micro-purchase procedures (e.g. purchase card) may be used.", "2.101 / 13.2")
        elif value <= t.simplified:
            add("Procedure", f"${value:,.0f} is above the micro-purchase threshold but within the simplified acquisition "
                f"threshold (${t.simplified:,.0f}): simplified acquisition procedures apply.", "2.101 / Part 13")
            add("Small business", "Must be set aside for small business unless the rule of two cannot be met "
                "(reasonable expectation of offers from two or more responsible small businesses).", "19.502-2 / 13.003(b)")
        else:
            if commercial and value <= t.commercial_simplified_cap:
                add("Procedure", f"Commercial acquisition ≤ ${t.commercial_simplified_cap / 1e6:.0f}M: simplified procedures "
                    "for commercial products/services are available.", "13.500")
            else:
                add("Procedure", "Above the simplified acquisition threshold: use FAR Part 15 (negotiation) or Part 14 "
                    "(sealed bidding)" + (" with Part 12 commercial procedures." if commercial else "."), "Part 15 / Part 12")
            add("Small business", "Set aside if the rule of two is met; otherwise document small business consideration.", "19.502-2")
            if not commercial:
                add("Publicizing", "Publish the notice at least 15 days before the solicitation and allow at least a "
                    "30-day response time.", "5.203")
        if sole and value > t.micro_purchase:
            if value <= t.ja_co_max:
                lvl = "contracting officer's certification serves as approval"
            elif value <= t.ja_competition_advocate_max:
                lvl = "competition advocate for the procuring activity (not delegable)"
            elif value <= t.ja_hca_max:
                lvl = "head of the procuring activity or a qualified designee"
            else:
                lvl = "senior procurement executive of the agency (not delegable)"
            add("Competition", "Sole source requires a written justification and approval (J&A) citing one of the "
                "6.302 authorities.", "6.303-1 / 6.303-2")
            add("J&A approval level", f"For ${value:,.0f}: {lvl}. (DoD, NASA and Coast Guard use a $150M "
                "ceiling instead of the civilian tier where applicable.)", "6.304")
        if value > t.certified_cost_pricing and not commercial and sole:
            add("Cost or pricing data", f"Above ${t.certified_cost_pricing / 1e6:.1f}M with no adequate price competition: "
                "certified cost or pricing data are required unless an exception applies.", "15.403-4 / 15.403-1")
        if services and re.search(r"(\d+)[- ]year", ql):
            yrs = int(re.search(r"(\d+)[- ]year", ql).group(1))
            if yrs > 5:
                add("Options", f"{yrs} years exceeds the 5-year limit on basic + option periods for services "
                    "(IT contracts excepted; check agency procedures).", "17.204(e)")
        return {"estimated_value": value, "commercial": commercial, "sole_source": sole, "services": services,
                "checklist": items, "thresholds_used": {k: v for k, v in self.t.__dict__.items() if k != "source"},
                "threshold_sources": self.t.source, "disclaimer": "Decision support only - not legal advice."}


# ------------------------------------------------------------------------------------------------
# Spending agent (NL -> SQL)
# ------------------------------------------------------------------------------------------------
SCHEMA = """awards(award_id TEXT, action_date TEXT, fiscal_year INT, awarding_agency TEXT, recipient_name TEXT,
  naics_code TEXT, naics_description TEXT, psc_code TEXT, obligation REAL, award_type TEXT,
  extent_competed TEXT, set_aside_type TEXT, number_of_offers INT, recipient_state TEXT)"""


def _authorizer(action, arg1, arg2, dbname, source):
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and arg1 == "awards":
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


AGENCY_ALIASES = {"dod": "Department of Defense", "pentagon": "Department of Defense", "hhs": "Department of Health and Human Services",
                  "va": "Department of Veterans Affairs", "dhs": "Department of Homeland Security",
                  "gsa": "General Services Administration", "doe": "Department of Energy",
                  "nasa": "National Aeronautics and Space Administration", "dot": "Department of Transportation",
                  "treasury": "Department of the Treasury", "doj": "Department of Justice"}


class SpendingAgent:

    def __init__(self, awards: pd.DataFrame, llm_sql=None):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        awards.to_sql("awards", self.db, index=False)
        self.db.set_authorizer(_authorizer)
        self.agencies = sorted(awards["awarding_agency"].unique())
        self.naics = dict(zip(awards["naics_description"].str.lower(), awards["naics_code"]))
        self.llm_sql = llm_sql

    def to_sql(self, q: str) -> str:
        if self.llm_sql:
            return self.llm_sql(q, SCHEMA)
        ql = q.lower()
        where = []
        fy = re.search(r"\b(?:fy|fiscal year)\s*(20\d{2})\b", ql)
        if fy:
            where.append(f"fiscal_year = {fy.group(1)}")
        agency = next((full for al, full in AGENCY_ALIASES.items() if re.search(rf"\b{al}\b", ql)), None)
        for a in self.agencies:
            short = re.sub(r"^department of (the )?", "", a.lower())
            if agency is None and (a.lower() in ql or re.search(rf"\b{re.escape(short)}\b", ql)):
                agency = a
        if agency in self.agencies:
            where.append(f"awarding_agency = '{agency}'")
        for desc, code in self.naics.items():
            if desc in ql:
                where.append(f"naics_code = '{code}'")
                break
        if "small business" in ql and "set" in ql:
            where.append("set_aside_type <> 'NONE'")
        w = (" WHERE " + " AND ".join(where)) if where else ""
        n = re.search(r"top (\d+)", ql)
        n = int(n.group(1)) if n else 10
        if re.search(r"sole.source|not competed", ql) and re.search(r"share|percent|%|proportion", ql):
            return ("SELECT ROUND(100.0 * SUM(CASE WHEN extent_competed = 'NOT COMPETED' THEN obligation ELSE 0 END) "
                    f"/ SUM(obligation), 2) AS sole_source_pct FROM awards{w}")
        if re.search(r"(vendors|recipients|contractors|companies)", ql):
            return f"SELECT recipient_name, ROUND(SUM(obligation), 2) AS total FROM awards{w} GROUP BY recipient_name ORDER BY total DESC LIMIT {n}"
        if re.search(r"by agency|which agenc|each agency", ql):
            return f"SELECT awarding_agency, ROUND(SUM(obligation), 2) AS total FROM awards{w} GROUP BY awarding_agency ORDER BY total DESC"
        if re.search(r"by fiscal year|per year|trend|each year", ql):
            return f"SELECT fiscal_year, ROUND(SUM(obligation), 2) AS total FROM awards{w} GROUP BY fiscal_year ORDER BY fiscal_year"
        if re.search(r"by (set.aside|set aside)", ql):
            return f"SELECT set_aside_type, ROUND(SUM(obligation), 2) AS total FROM awards{w} GROUP BY set_aside_type ORDER BY total DESC"
        if re.search(r"how many|number of", ql):
            return f"SELECT COUNT(*) AS n_awards FROM awards{w}"
        if re.search(r"average number of offers|avg(erage)? offers", ql):
            return f"SELECT ROUND(AVG(number_of_offers), 2) AS avg_offers FROM awards{w}"
        return f"SELECT ROUND(SUM(obligation), 2) AS total_obligations FROM awards{w}"

    def run(self, q: str) -> dict:
        sql = self.to_sql(q)
        try:
            df = pd.read_sql_query(sql, self.db)
            return {"sql": sql, "rows": df.to_dict(orient="records"), "error": None}
        except Exception as e:
            return {"sql": sql, "rows": [], "error": f"blocked or invalid SQL: {e}"}


# ------------------------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------------------------
class Orchestrator:
    def __init__(self, retriever, scope, sections: dict[str, Section], awards: pd.DataFrame, generator=None):
        self.retriever, self.scope, self.gen = retriever, scope, generator
        self.compliance = ComplianceAgent(sections)
        self.spending = SpendingAgent(awards)

    def ask(self, q: str) -> dict:
        intent = route(q)
        if intent == "spending":
            return {"intent": intent, **self.spending.run(q)}
        if intent == "compliance":
            out = self.compliance.check(q)
            ans = answer(q, self.retriever, self.scope, self.gen)
            return {"intent": intent, **out, "supporting_answer": ans.text, "citations": ans.citations}
        a = answer(q, self.retriever, self.scope, self.gen)
        return {"intent": intent, "answer": a.text, "citations": a.citations, "abstained": a.abstained,
                "confidence": a.confidence, "backend": a.backend, "guardrail_notes": a.guardrail_notes,
                "sources": [{"sid": h.chunk.sid, "title": h.chunk.title, "score": round(h.score, 3)} for h in a.hits]}
