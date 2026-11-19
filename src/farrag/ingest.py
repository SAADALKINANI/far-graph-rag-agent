"""Parse the official FAR (GSA/GSA-Acquisition-FAR, HTML edition) into sections, chunks and a
citation graph.

* Every ``<article id="FAR_15_404_4">`` is one regulatory unit (part / subpart / section /
  subsection). We keep only its *own* text (nested child articles are separate units).
* Chunks are paragraph-aligned (~220 words) and carry a breadcrumb header
  ("FAR 15.404-4 Profit > Part 15 Contracting by Negotiation") so each chunk is
  self-describing for both retrieval and the LLM.
* Cross-references ("see 15.404-1(b)", "subpart 19.8", "part 12", "52.212-4") become edges
  in a citation graph used for Graph-RAG expansion.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx
from bs4 import BeautifulSoup

SEC_RE = re.compile(r"\b(\d{1,2}\.\d{3,4}(?:-\d{1,3})?)(?=[\s(,.;:)]|$)")
SUBPART_RE = re.compile(r"\b[Ss]ubpart\s+(\d{1,2}\.\d{1,2})\b")
PART_RE = re.compile(r"\b[Pp]art\s+(\d{1,2})\b(?!\.)")


@dataclass
class Section:
    sid: str                    # "15.404-4", "Subpart 15.4", "Part 15"
    title: str
    level: str                  # part | subpart | section
    part: int
    parent: str | None
    text: str
    cites: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    cid: str
    sid: str
    title: str
    header: str
    text: str


def _norm_id(article_id: str) -> tuple[str, str] | None:
    m = re.match(r"FAR_Part_(\d+)$", article_id)
    if m:
        return f"Part {m.group(1)}", "part"
    m = re.match(r"FAR_Subpart_(\d+)_(\d+)$", article_id)
    if m:
        return f"Subpart {m.group(1)}.{m.group(2)}", "subpart"
    m = re.match(r"FAR_(\d+)_(\d+)(?:_(\d+))?$", article_id)
    if m:
        sec = f"{m.group(1)}.{m.group(2)}" + (f"-{m.group(3)}" if m.group(3) else "")
        return sec, "section"
    return None


def _own_text(article) -> str:
    parts = []
    for el in article.find_all(["p", "li", "td", "pre"], recursive=True):
        if el.find_parent("article") is not article:
            continue
        if el.find_parent(["p", "li", "td"]) is not None and el.name != "td":
            continue
        t = " ".join(el.get_text(" ", strip=True).split())
        if t:
            parts.append(t)
    return "\n".join(parts)


def parse_far(html_dir: str | Path) -> list[Section]:
    sections: dict[str, Section] = {}
    for f in sorted(Path(html_dir).glob("FAR_Part_*.html"), key=lambda p: int(p.stem.split("_")[-1])):
        soup = BeautifulSoup(f.read_text(encoding="utf-8", errors="ignore"), "lxml")
        part_no = int(f.stem.split("_")[-1])
        for art in soup.find_all("article"):
            ident = _norm_id(art.get("id", ""))
            if not ident:
                continue
            sid, level = ident
            h = art.find(re.compile(r"^h\d$"))
            title = h.get_text(" ", strip=True) if h else sid
            title = re.sub(r"^\s*(Part|Subpart)?\s*[\d.\-]+\s*[-–]?\s*", "", title).strip() or title
            title = " ".join(title.split())
            parent_art = art.find_parent("article")
            parent = _norm_id(parent_art.get("id", ""))[0] if parent_art and _norm_id(parent_art.get("id", "")) else None
            text = _own_text(art)
            if level == "part" or (level == "subpart" and len(text) < 40):
                text = ""   # tables of contents - not useful content
            if sid in sections:
                continue
            sections[sid] = Section(sid, title, level, part_no, parent, text)
    known = set(sections)
    for s in sections.values():
        cites = set()
        for m in SEC_RE.findall(s.text):
            base = m if m in known else m.split("-")[0]
            if base in known and base != s.sid:
                cites.add(base)
        for m in SUBPART_RE.findall(s.text):
            if f"Subpart {m}" in known:
                cites.add(f"Subpart {m}")
        for m in PART_RE.findall(s.text):
            if f"Part {m}" in known and int(m) != s.part:
                cites.add(f"Part {m}")
        s.cites = sorted(cites)
    return list(sections.values())


def chunk_sections(sections: list[Section], target_words: int = 220) -> list[Chunk]:
    by_id = {s.sid: s for s in sections}
    chunks = []
    for s in sections:
        if not s.text:
            continue
        crumbs, p = [], s.parent
        while p and p in by_id:
            crumbs.append(f"{p} {by_id[p].title}")
            p = by_id[p].parent
        header = f"FAR {s.sid} {s.title}" + (" > " + " > ".join(crumbs) if crumbs else "")
        if s.sid in DEFINITION_SECTIONS:          # one chunk per defined term -> precise definition lookup
            chunks.extend(_definition_chunks(s, header))
            continue
        buf, n, k = [], 0, 0
        paras = s.text.split("\n")
        for i, para in enumerate(paras):
            w = len(para.split())
            buf.append(para); n += w
            if n >= target_words or i == len(paras) - 1:
                chunks.append(Chunk(f"{s.sid}#{k}", s.sid, s.title, header, "\n".join(buf)))
                buf, n, k = [], 0, k + 1
    return chunks


DEFINITION_SECTIONS = {"2.101"}
TERM_RE = re.compile(r"^([A-Z][A-Za-z0-9 ,()/'’\-]{2,90}?)(?:,? as used in [^,]+,)? means\b")


def _definition_chunks(s: Section, header: str) -> list[Chunk]:
    out, cur_term, buf, k = [], None, [], 0

    def flush():
        nonlocal k
        if buf:
            title = cur_term or s.title
            out.append(Chunk(f"{s.sid}#{k}", s.sid, title, f"FAR {s.sid} Definitions: {title}", "\n".join(buf)))
            k += 1

    for para in s.text.split("\n"):
        m = TERM_RE.match(para)
        if m:
            flush(); cur_term, buf = m.group(1).strip(), [para]
        else:
            buf.append(para)
            if len(" ".join(buf).split()) > 400:      # very long definitions get split
                flush(); buf = []
    flush()
    return out


def citation_graph(sections: list[Section]) -> nx.DiGraph:
    G = nx.DiGraph()
    for s in sections:
        G.add_node(s.sid, title=s.title, level=s.level, part=s.part)
    for s in sections:
        if s.parent:
            G.add_edge(s.sid, s.parent, kind="parent")
            G.add_edge(s.parent, s.sid, kind="child")
        for c in s.cites:
            G.add_edge(s.sid, c, kind="cites")
    return G


def build_corpus(html_dir: str | Path, out_dir: str | Path) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    secs = parse_far(html_dir)
    chunks = chunk_sections(secs)
    G = citation_graph(secs)
    (out / "sections.jsonl").write_text("\n".join(json.dumps(asdict(s)) for s in secs))
    (out / "chunks.jsonl").write_text("\n".join(json.dumps(asdict(c)) for c in chunks))
    nx.write_edgelist(G, out / "citation_graph.edgelist", data=["kind"])
    stats = {"sections": len(secs), "with_text": sum(bool(s.text) for s in secs), "chunks": len(chunks),
             "citation_edges": sum(1 for *_, d in G.edges(data=True) if d["kind"] == "cites"),
             "words": sum(len(c.text.split()) for c in chunks)}
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_corpus(corpus_dir: str | Path):
    d = Path(corpus_dir)
    secs = [Section(**json.loads(line)) for line in (d / "sections.jsonl").read_text().splitlines()]
    chunks = [Chunk(**json.loads(line)) for line in (d / "chunks.jsonl").read_text().splitlines()]
    return secs, chunks, citation_graph(secs)
