import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farrag.agents import ComplianceAgent, SpendingAgent, route  # noqa: E402
from farrag.evaluate import section_hit  # noqa: E402
from farrag.generate import CITE_RE, ExtractiveGenerator, answer  # noqa: E402
from farrag.ingest import Section, chunk_sections, citation_graph  # noqa: E402
from farrag.retrieval import Retriever, tokenize  # noqa: E402
from farrag.spending_data import synthetic_awards  # noqa: E402

DEFS = ("Micro-purchase threshold means $15,000, except for construction.\n"
        "Simplified acquisition threshold means $350,000, except for contingency operations.\n"
        "Commercial product means a product customarily used by the general public.")
SECS = [
    Section("Part 2", "Definitions", "part", 2, None, ""),
    Section("2.101", "Definitions.", "section", 2, "Part 2", DEFS),
    Section("13.500", "General.", "section", 13, None, "Simplified procedures not exceeding $9 million for commercial products. See 2.101."),
    Section("15.403-4", "Requiring certified cost or pricing data.", "section", 15, None,
            "The threshold for obtaining certified cost or pricing data is $950,000 for prime contracts awarded before "
            "July 1, 2018, and $2.5 million for prime contracts awarded on or after July 1, 2018.", ["2.101"]),
    Section("6.304", "Approval of the justification.", "section", 6, None,
            "(1) For a proposed contract not exceeding $900,000, the contracting officer certification serves as approval. "
            "(2) For a proposed contract over $900,000 but not exceeding $20 million, by the advocate for competition. "
            "(3) For a proposed contract over $20 million, but not exceeding $90 million, by the head of the procuring activity."),
    Section("17.204", "Contracts.", "section", 17, None,
            "The total of the basic and option periods shall not exceed 5 years in the case of services."),
    Section("31.205-51", "Costs of alcoholic beverages.", "section", 31, None, "Costs of alcoholic beverages are unallowable."),
]


@pytest.fixture(scope="module")
def retr():
    chunks = chunk_sections(SECS)
    from farrag.retrieval import LSAEmbedder
    return Retriever(SECS, chunks, citation_graph(SECS), embedder=LSAEmbedder(dim=4))


def test_definitions_are_chunked_per_term():
    titles = [c.title for c in chunk_sections(SECS) if c.sid == "2.101"]
    assert "Micro-purchase threshold" in titles and "Commercial product" in titles


def test_tokenizer_keeps_far_citations():
    assert "15.403-4" in tokenize("see FAR 15.403-4 for details")


def test_retrieval_finds_exact_section(retr):
    for mode in ("bm25", "hybrid", "graph", "full"):
        hits = retr.search("Are alcoholic beverages allowable costs?", k=3, mode=mode)
        assert hits[0].chunk.sid == "31.205-51"


def test_explicit_citation_in_query_is_resolved(retr):
    hits = retr.search("What does FAR 17.204 say?", k=3, mode="graph")
    assert "17.204" in [h.chunk.sid for h in hits]


def test_extractive_answer_always_cites_retrieved_sections(retr):
    a = answer("What is the micro-purchase threshold?", retr, scope=None, gen=ExtractiveGenerator())
    cites = CITE_RE.findall(a.text)
    assert cites and set(cites) <= {h.chunk.sid for h in a.hits} and "$15,000" in a.text


def test_llm_citation_guardrail_replaces_hallucinated_citations(retr):
    class FakeLLM:
        name = "fake"
        def __call__(self, q, hits):
            return "The threshold is $1 [FAR 99.999]."
    a = answer("What is the micro-purchase threshold?", retr, scope=None, gen=FakeLLM())
    assert "99.999" not in a.text and any("guardrail" in n for n in a.guardrail_notes)


def test_compliance_thresholds_parsed_from_text():
    ca = ComplianceAgent({s.sid: s for s in SECS})
    assert ca.t.micro_purchase == 15000 and ca.t.simplified == 350000 and ca.t.certified_cost_pricing == 2.5e6
    out = ca.check("We plan a $3 million sole-source services contract for 7 years")
    topics = {i["topic"] for i in out["checklist"]}
    assert {"Competition", "J&A approval level", "Cost or pricing data", "Options"} <= topics


def test_router():
    assert route("What is the simplified acquisition threshold?") == "regulation"
    assert route("Top 5 vendors by obligations in FY2024") == "spending"
    assert route("We plan to buy $40,000 of furniture, what do we need?") == "compliance"


def test_sql_agent_is_read_only():
    ag = SpendingAgent(synthetic_awards(500))
    assert ag.run("How many contracts did GSA award in FY2022?")["error"] is None
    with pytest.raises(Exception):
        ag.db.execute("DELETE FROM awards")


def test_section_hit_handles_subparts():
    assert section_hit("19.805-1", ["Subpart 19.8"]) and not section_hit("19.1305", ["Subpart 19.1"])
    assert section_hit("19.1305", ["Subpart 19.13"])
