# memory.py
"""
Persistent memory store using SQLite (+ FTS5 if available).
Simple API:
  mem = MemoryStore(db_path="~/hero_memory.db")
  mem.add("user:123", "some text")
  mem.search("query", limit=6) -> list of tuples (id, source, created_ts, snippet_or_content)
  mem.count()
  mem.ingest_url(url)  # optional (requires requests + beautifulsoup4)
"""
from __future__ import annotations
import sqlite3
import time
import os
from typing import List, Tuple, Optional

# optional fetch
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

DEFAULT_DB = os.path.expanduser("~/hero_memory.db")


class MemoryStore:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = os.path.expanduser(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self):
        cur = self._conn.cursor()
        # docs: canonical table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY,
                source TEXT,
                content TEXT,
                created_ts REAL
            )
            """
        )
        # try to create FTS5; if host sqlite doesn't support, this may raise — ignore then fallback to LIKE-based search
        try:
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(content, source, content='docs', content_rowid='id');")
        except Exception:
            # FTS not available; we'll still keep docs table and fallback to LIKE queries
            pass
        self._conn.commit()

    def add(self, source: str, content: str) -> int:
        ts = time.time()
        cur = self._conn.cursor()
        cur.execute("INSERT INTO docs (source, content, created_ts) VALUES (?, ?, ?)", (source, content, ts))
        rowid = cur.lastrowid
        # If FTS present, also insert into the mirror FTS table (works on many builds)
        try:
            cur.execute("INSERT INTO docs_fts (rowid, content, source) VALUES (?, ?, ?)", (rowid, content, source))
        except Exception:
            # ignore if docs_fts not present
            pass
        self._conn.commit()
        return rowid

    def search(self, q: str, limit: int = 6) -> List[Tuple[int, str, float, str]]:
        """
        Return rows: (id, source, created_ts, snippet_or_content)
        Uses FTS MATCH if available else falls back to LIKE on docs.content.
        """
        cur = self._conn.cursor()
        q = (q or "").strip()
        if not q:
            return []

        # Try FTS MATCH approach
        try:
            cur.execute(
                """
                SELECT d.id, d.source, d.created_ts,
                       snippet(docs_fts, 0, '[', ']', '...', 120) as snippet
                  FROM docs_fts JOIN docs d ON docs_fts.rowid = d.id
                 WHERE docs_fts MATCH ?
                 LIMIT ?
                """,
                (q, limit),
            )
            rows = cur.fetchall()
            if rows:
                return rows
        except Exception:
            # Fallback to LIKE
            pass

        # Fallback: simple LIKE search
        like_q = f"%{q}%"
        cur.execute("SELECT id, source, created_ts, substr(content,1,240) FROM docs WHERE content LIKE ? ORDER BY created_ts DESC LIMIT ?", (like_q, limit))
        return cur.fetchall()

    def count(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM docs")
        return int(cur.fetchone()[0] or 0)

    def ingest_url(self, url: str, source_label: Optional[str] = None) -> Optional[int]:
        """
        Fetch a URL and extract text content to store. Requires requests + beautifulsoup4.
        Returns inserted row id or None.
        """
        if requests is None or BeautifulSoup is None:
            raise RuntimeError("ingest_url requires requests and beautifulsoup4: pip install requests beautifulsoup4")
        try:
            r = requests.get(url, timeout=12, headers={"User-Agent": "HeroXMemory/1.0"})
            if r.status_code != 200:
                return None
            html = r.text
            soup = BeautifulSoup(html, "html.parser")
            # remove some noisy parts
            for s in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "iframe"]):
                s.decompose()
            title = (soup.title.string.strip() if soup.title and soup.title.string else "")
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(" ", strip=True)]
            text = "\n\n".join(paragraphs)
            if not text.strip():
                text = soup.get_text(" ", strip=True)
            content = f"URL: {url}\nTitle: {title}\n\n{text}".strip()
            src = source_label or f"url:{url}"
            return self.add(src, content)
        except Exception:
            return None

    def close(self):
        try:
            self._conn.commit()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
