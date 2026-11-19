"""Contract-awards table in the USAspending.gov schema.

``fetch_usaspending`` documents how to pull real award data (api.usaspending.gov, no key needed);
``synthetic_awards`` produces a clearly-labelled synthetic sample with the same columns so the
NL->SQL agent and its evaluation run fully offline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGENCIES = ["Department of Defense", "Department of Health and Human Services", "Department of Veterans Affairs",
            "Department of Homeland Security", "General Services Administration", "Department of Energy",
            "National Aeronautics and Space Administration", "Department of Transportation",
            "Department of the Treasury", "Department of Justice"]
AG_W = np.array([45, 12, 9, 7, 5, 6, 4, 3, 3, 3], float)
NAICS = {"541512": "computer systems design services", "541511": "custom computer programming services",
         "541330": "engineering services", "561720": "janitorial services",
         "236220": "commercial and institutional building construction", "541611": "management consulting services",
         "336411": "aircraft manufacturing", "325412": "pharmaceutical preparation manufacturing",
         "541715": "research and development in the physical sciences", "518210": "data processing and hosting services"}
COMPETED = ["FULL AND OPEN COMPETITION", "FULL AND OPEN COMPETITION AFTER EXCLUSION OF SOURCES", "NOT COMPETED",
            "NOT AVAILABLE FOR COMPETITION"]
SET_ASIDES = ["NONE", "SBA", "8A", "HZC", "SDVOSBC", "WOSB"]


def synthetic_awards(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pre = ["Apex", "Liberty", "Summit", "Patriot", "Vector", "Keystone", "Meridian", "Sentinel", "Frontier", "Atlas",
           "Pinnacle", "Horizon", "Ironclad", "Beacon", "Cardinal", "Granite", "Northstar", "Redwood", "Trident", "Vanguard"]
    suf = ["Federal Solutions LLC", "Systems Inc", "Technologies Corp", "Consulting Group", "Defense Services",
           "Analytics LLC", "Engineering Inc", "Health Partners"]
    recips = [f"{a} {b}" for a in pre for b in suf]
    rw = rng.pareto(1.2, len(recips)) + 0.1
    fy = rng.choice([2021, 2022, 2023, 2024, 2025], n, p=[.17, .19, .2, .22, .22])
    naics = rng.choice(list(NAICS), n)
    sa = rng.choice(SET_ASIDES, n, p=[.62, .16, .08, .04, .06, .04])
    comp = np.where(sa != "NONE", COMPETED[1], rng.choice([COMPETED[0], COMPETED[2], COMPETED[3]], n, p=[.7, .25, .05]))
    obl = np.round(rng.lognormal(12.3, 1.6, n) * np.where(naics == "336411", 8, 1), 2)
    month = rng.integers(1, 13, n)
    return pd.DataFrame({
        "award_id": [f"SYN-{i:07d}" for i in range(n)],
        "action_date": [f"{y - (1 if m >= 10 else 0)}-{m:02d}-15" for y, m in zip(fy, month)],
        "fiscal_year": fy, "awarding_agency": rng.choice(AGENCIES, n, p=AG_W / AG_W.sum()),
        "recipient_name": rng.choice(recips, n, p=rw / rw.sum()), "naics_code": naics,
        "naics_description": [NAICS[c] for c in naics], "psc_code": rng.choice(["D399", "R425", "S201", "J099", "AJ11"], n),
        "obligation": obl, "award_type": rng.choice(["DEFINITIVE CONTRACT", "DELIVERY ORDER", "PURCHASE ORDER", "BPA CALL"], n),
        "extent_competed": comp, "set_aside_type": sa,
        "number_of_offers": np.where(np.isin(comp, COMPETED[2:]), 1, rng.integers(1, 12, n)),
        "recipient_state": rng.choice(["VA", "MD", "DC", "TX", "CA", "FL", "CO", "AL"], n)})


FETCH_DOC = """Real data: POST https://api.usaspending.gov/api/v2/bulk_download/awards/ with filters
{"prime_award_types": ["A","B","C","D"], "date_type": "action_date", "date_range": {...}, "agencies": [...]}
then load the resulting CSV and rename columns to the schema in agents.SCHEMA
(e.g. 'awarding_agency_name' -> awarding_agency, 'recipient_name' -> recipient_name,
'federal_action_obligation' -> obligation, 'extent_competed' -> extent_competed,
'type_of_set_aside' -> set_aside_type, 'number_of_offers_received' -> number_of_offers)."""
