"""Federal Acquisition Assistant.  Run:  streamlit run app.py   (after scripts/run_pipeline.py)

Set ANTHROPIC_API_KEY or OPENAI_API_KEY / OPENAI_BASE_URL (e.g. a local Ollama) to switch the
answer generator from the offline extractive mode to an LLM; citation guardrails apply either way.
"""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from farrag.agents import Orchestrator  # noqa: E402
from farrag.spending_data import synthetic_awards  # noqa: E402

st.set_page_config(page_title="Federal Acquisition Assistant", layout="wide")


@st.cache_resource
def boot():
    b = pickle.load(open(ROOT / "artifacts" / "retriever.pkl", "rb"))
    r = b["retriever"]
    return Orchestrator(r, b["scope"], r.sections, synthetic_awards()), r


orch, retr = boot()
M = json.loads((ROOT / "results" / "metrics.json").read_text())
tab_chat, tab_eval = st.tabs(["Ask", "Evaluation scorecard"])
with tab_chat:
    st.title("Federal Acquisition Assistant")
    st.caption("Graph-RAG over the full FAR · compliance checklist agent · spending analytics (SQL) · every claim cited. "
               "Decision support only - not legal advice.")
    examples = ["What is the micro-purchase threshold?",
                "We plan to award a $3.2 million sole-source software maintenance services contract for 7 years, what do we need?",
                "Top 5 vendors by obligations in FY2024", "Who approves a sole-source justification between $900,000 and $20 million?"]
    q = st.text_input("Question", examples[0])
    st.caption("Try: " + " · ".join(f"`{e}`" for e in examples[1:]))
    if q:
        out = orch.ask(q)
        st.markdown(f"**Routed to:** `{out['intent']}` agent")
        if out["intent"] == "spending":
            st.code(out["sql"], language="sql")
            st.dataframe(pd.DataFrame(out["rows"]))
            st.caption("Data: synthetic sample in the USAspending.gov schema (see README to load real awards).")
        elif out["intent"] == "compliance":
            st.dataframe(pd.DataFrame(out["checklist"]), use_container_width=True)
            st.caption("Thresholds were parsed from the current FAR text at start-up: " +
                       ", ".join(f"{k} = ${v:,.0f}" for k, v in out["thresholds_used"].items()))
            st.markdown("**Supporting FAR text:** " + out["supporting_answer"])
        else:
            st.markdown(out["answer"])
            st.caption(f"backend: {out['backend']} · in-scope confidence {out['confidence']:.2f} · " + "; ".join(out["guardrail_notes"]))
            with st.expander("Retrieved sources"):
                for s in out["sources"]:
                    st.write(f"FAR {s['sid']} - {s['title']} (score {s['score']})")
with tab_eval:
    st.markdown((ROOT / "results" / "scorecard.md").read_text())
    st.image(str(ROOT / "results" / "figures" / "retrieval_ablation.png"))
