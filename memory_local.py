# memory_local.py
from __future__ import annotations
import sqlite3
import time
import hashlib
import os
from typing import List, Tuple, Optional

DEFAULT_DB = "hero_memory.db"

class MemoryLocal:
    """
    Simple SQLite-backed local memory store with FTS5 if available.
    Stores chunks with metadata and allows full-text search (or LIKE fallback).
    """

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self._conn.cursor()
        # main table for metadata
        cur.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            hash TEXT UNIQUE,
            title TEXT,
            source TEXT,
            url TEXT,
            ts INTEGER
        )""")
        # try to create FTS5 virtual table; if not available, we rely on fallback search
        try:
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(content, title, url, source, content='')")
            # link content via shadow table insertions - we'll manage items_fts ourselves
        except Exception:
            # FTS not available — that's OK; fallback search will use LIKE on a separate table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS items_text (
                item_id INTEGER PRIMARY KEY,
                content TEXT,
                FOREIGN KEY(item_id) REFERENCES items(id) ON DELETE CASCADE
            )""")
        self._conn.commit()

    def _compute_hash(self, text: str) -> str:
        h = hashlib.sha256()
        h.update(text.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def add_chunk(self, content: str, title: str = "", source: str = "", url: str = "", ts: Optional[int] = None) -> Optional[int]:
        """
        Add a chunk. Dedup by content-hash. Returns item id or None if duplicate.
        """
        if ts is None:
            ts = int(time.time())
        h = self._compute_hash(content)
        cur = self._conn.cursor()
        # check duplicate
        cur.execute("SELECT id FROM items WHERE hash = ?", (h,))
        row = cur.fetchone()
        if row:
            return None
        # insert metadata
        cur.execute("INSERT INTO items (hash, title, source, url, ts) VALUES (?, ?, ?, ?, ?)",
                    (h, title[:400], source[:200], url[:800], ts))
        item_id = cur.lastrowid
        # insert content into FTS or fallback
        try:
            cur.execute("INSERT INTO items_fts (rowid, content, title, url, source) VALUES (?, ?, ?, ?, ?)",
                        (item_id, content, title, url, source))
        except Exception:
            # fallback: insert into items_text
            cur.execute("INSERT INTO items_text (item_id, content) VALUES (?, ?)", (item_id, content))
        self._conn.commit()
        return item_id

    def search(self, query: str, limit: int = 6) -> List[Tuple[int, str, int, str]]:
        """
        Search memory for `query`.
        Returns list of tuples: (id, source, ts, snippet)
        """
        cur = self._conn.cursor()
        q = query.strip()
        # try FTS5 match first
        try:
            cur.execute("SELECT rowid as id, source, ts, snippet(items_fts, 0, '[', ']', '...', 64) as snippet FROM items_fts WHERE items_fts MATCH ? LIMIT ?",
                        (q, limit))
            rows = cur.fetchall()
            if rows:
                return [(r["id"], r["source"], r["ts"], r["snippet"]) for r in rows]
        except Exception:
            pass

        # fallback: LIKE search on title/url/source or items_text
        like = f"%{q}%"
        try:
            cur.execute("""
                SELECT i.id, i.source, i.ts,
                       substr(t.content, 1, 220) as snippet
                FROM items i
                JOIN items_text t ON t.item_id = i.id
                WHERE t.content LIKE ?
                   OR i.title LIKE ?
                   OR i.source LIKE ?
                   OR i.url LIKE ?
                ORDER BY i.ts DESC
                LIMIT ?
            """, (like, like, like, like, limit))
            rows = cur.fetchall()
            return [(r["id"], r["source"], r["ts"], r["snippet"]) for r in rows]
        except Exception:
            return []

    def count(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) as c FROM items")
        return int(cur.fetchone()["c"] or 0)

    def get(self, item_id: int) -> Optional[dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT i.*, coalesce(t.content, '') as content FROM items i LEFT JOIN items_text t ON t.item_id = i.id WHERE i.id = ?", (item_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "title": row["title"],
            "source": row["source"],
            "url": row["url"],
            "ts": row["ts"],
            "content": row["content"],
        }

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass
