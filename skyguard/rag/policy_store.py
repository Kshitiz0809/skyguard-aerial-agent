"""Local RAG over the site SOP and flight manual.

Each numbered clause (e.g. ``4.4`` or ``C.2``) becomes one chunk, so a
retrieved result can be cited precisely ("SOP-SUB-B-v1.3 §4.4").

Backends:
* ``chroma``  – ChromaDB persistent collection with its built-in
                all-MiniLM-L6-v2 ONNX embedder (downloads ~80 MB once).
* ``keyword`` – zero-dependency BM25-style fallback, used in tests/offline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CLAUSE_RE = re.compile(r"^(?P<id>(?:\d+|[A-Z])\.\d+)\s+(?P<text>.+)$")
DOC_ID_RE = re.compile(r"Document ID:\s*([A-Za-z0-9.\-]+)")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "at", "for", "is", "be", "must", "may",
    "with", "by", "any", "all", "not", "do", "it", "this", "that", "are", "as", "from", "than", "per",
}


@dataclass
class PolicyChunk:
    chunk_id: str
    doc_id: str
    clause: str
    section: str
    text: str
    source: str
    score: float = 0.0

    @property
    def citation(self) -> str:
        return f"{self.doc_id} §{self.clause}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["citation"] = self.citation
        return d


def chunk_markdown(path: Path) -> list[PolicyChunk]:
    content = path.read_text(encoding="utf-8")
    doc_match = DOC_ID_RE.search(content)
    doc_id = doc_match.group(1) if doc_match else path.stem
    chunks: list[PolicyChunk] = []
    section = ""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        m = CLAUSE_RE.match(line)
        if m:
            clause = m.group("id")
            chunks.append(
                PolicyChunk(
                    chunk_id=f"{doc_id}:{clause}",
                    doc_id=doc_id,
                    clause=clause,
                    section=section,
                    text=m.group("text").strip(),
                    source=path.name,
                )
            )
    return chunks


def _tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class _KeywordIndex:
    """Tiny BM25 implementation (no dependencies)."""

    def __init__(self, chunks: list[PolicyChunk], k1: float = 1.4, b: float = 0.75):
        self.chunks = chunks
        self.docs = [_tokens(f"{c.section} {c.text}") for c in chunks]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def query(self, text: str, k: int) -> list[PolicyChunk]:
        q = _tokens(text)
        scored = []
        for chunk, doc in zip(self.chunks, self.docs):
            tf = Counter(doc)
            score = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, chunk in scored[:k]:
            c = PolicyChunk(**{**asdict(chunk), "score": round(score, 3)})
            out.append(c)
        return out


class PolicyStore:
    COLLECTION = "skyguard_policies"

    def __init__(self, sop_dir: Path, persist_dir: Path | None = None, backend: str = "auto"):
        self.sop_dir = Path(sop_dir)
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.chunks: list[PolicyChunk] = []
        for path in sorted(self.sop_dir.glob("*.md")):
            self.chunks.extend(chunk_markdown(path))
        if not self.chunks:
            raise FileNotFoundError(f"No policy clauses found in {self.sop_dir}")
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self.backend = self._init_backend(backend)

    # ------------------------------------------------------------------ init
    def _init_backend(self, backend: str) -> str:
        if backend in {"auto", "chroma"}:
            try:
                self._init_chroma()
                return "chroma"
            except Exception as exc:  # chromadb missing or embedder download failed
                if backend == "chroma":
                    raise
                self._chroma_error = repr(exc)
        self._keyword = _KeywordIndex(self.chunks)
        return "keyword"

    def _init_chroma(self) -> None:
        import chromadb

        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.persist_dir))
        else:
            client = chromadb.EphemeralClient()
        self._collection = client.get_or_create_collection(self.COLLECTION, metadata={"hnsw:space": "cosine"})
        existing = set(self._collection.get(include=[])["ids"])
        wanted = {c.chunk_id for c in self.chunks}
        if existing != wanted:
            if existing:
                self._collection.delete(ids=list(existing))
            self._collection.add(
                ids=[c.chunk_id for c in self.chunks],
                documents=[f"[{c.section}] {c.text}" for c in self.chunks],
                metadatas=[{"doc_id": c.doc_id, "clause": c.clause, "section": c.section, "source": c.source} for c in self.chunks],
            )

    # ----------------------------------------------------------------- query
    def query(self, text: str, k: int = 4) -> list[PolicyChunk]:
        if self.backend == "chroma":
            res = self._collection.query(query_texts=[text], n_results=min(k, len(self.chunks)))
            out = []
            for cid, dist in zip(res["ids"][0], res["distances"][0]):
                base = self._by_id[cid]
                out.append(PolicyChunk(**{**asdict(base), "score": round(1 - float(dist), 3)}))
            return out
        return self._keyword.query(text, k)

    def multi_query(self, queries: list[str], k: int = 4) -> list[PolicyChunk]:
        """Run several queries and merge results, keeping each clause's best score."""
        best: dict[str, PolicyChunk] = {}
        for q in queries:
            for c in self.query(q, k):
                if c.chunk_id not in best or c.score > best[c.chunk_id].score:
                    best[c.chunk_id] = c
        return sorted(best.values(), key=lambda c: c.score, reverse=True)

    def get(self, doc_prefix: str, clause: str) -> PolicyChunk | None:
        for c in self.chunks:
            if c.doc_id.startswith(doc_prefix) and c.clause == clause:
                return c
        return None
