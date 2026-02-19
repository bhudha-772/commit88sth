# ingest.py (only change: use get_soup that prefers lxml then falls back)
from __future__ import annotations
import requests
from bs4 import BeautifulSoup
import time
from typing import Callable, List, Optional
from urllib.parse import urlparse, urljoin
import urllib.robotparser
import os

# chunk config
MAX_CHUNK_CHARS = 800
CHUNK_OVERLAP = 100

def get_soup(html: str):
    """
    Return a BeautifulSoup object. Prefer lxml if available, otherwise fall
    back to the built-in 'html.parser'.
    """
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        # lxml not available or failed to parse — fallback
        return BeautifulSoup(html, "html.parser")

def _clean_html_text(html: str) -> str:
    soup = get_soup(html)
    for s in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        s.decompose()
    parts = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]):
        txt = tag.get_text(separator=" ", strip=True)
        if txt:
            parts.append(txt)
    text = "\n\n".join(parts).strip()
    if not text:
        text = soup.get_text(separator=" ", strip=True)
    return text


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks = []
    cur = ""
    for p in paragraphs:
        cand = (cur + "\n\n" + p) if cur else p
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                chunks.append(cur.strip())
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = start + max_chars
                    chunks.append(p[start:end].strip())
                    start = end - overlap if end - overlap > start else end
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur.strip())
    return chunks

def _robots_ok(url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = urljoin(base, "/robots.txt")
    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # if robots can't be read, allow by default (more permissive)
        return True

def _fetch_html(url: str, headers: dict, timeout: int = 15, allow_redirects: bool = True) -> dict:
    """
    Fetch URL, return dict {ok, status_code, ctype, text, error}
    """
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=allow_redirects)
        return {"ok": True, "status": r.status_code, "ctype": r.headers.get("Content-Type", ""), "text": r.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def ingest_url(
    url: str,
    memory,
    progress: Optional[Callable[[str], None]] = None,
    max_chunks: int = 64,
    ignore_robots: bool = False,
) -> dict:
    """
    Fetch URL, extract text, split into chunks, store into memory.
    - memory is expected to expose add_chunk(text, title=..., source=..., url=..., ts=...)
    - progress(msg) will be called if provided.
    - ignore_robots: if True, skip robots.txt check (useful for trusted seeds like Wikipedia).
    Returns a dict with {ok:bool, added:int, total_candidates:int, ids:list, title:... } or error.
    """
    if progress:
        progress(f"Checking robots.txt for {url} ...")

    parsed = urlparse(url)
    netloc = parsed.netloc or ""

    # Auto-allow wikipedia seeds (they are fine for ingesting public content)
    if "wikipedia.org" in netloc.lower():
        ignore_robots = True

    if not ignore_robots:
        try:
            ok = _robots_ok(url)
        except Exception:
            ok = True
        if not ok:
            msg = "Robots.txt disallows fetching this URL."
            if progress:
                progress(msg)
            return {"ok": False, "error": "robots.txt disallows", "reason": msg}

    headers = {"User-Agent": "HeroX/1.0 (+https://example.local)", "Accept": "text/html,*/*;q=0.9"}
    if progress:
        progress("Fetching URL ...")

    fetched = _fetch_html(url, headers=headers)
    if not fetched.get("ok"):
        # try wiki render fallback or return error
        if progress:
            progress(f"Fetch failed: {fetched.get('error')}. Trying action=render fallback ...")
        # attempt action=render fallback
        sep = "&" if "?" in url else "?"
        render_url = url + sep + "action=render"
        fetched = _fetch_html(render_url, headers=headers)
        if not fetched.get("ok"):
            if progress:
                progress(f"Render fallback failed: {fetched.get('error')}")
            return {"ok": False, "error": f"fetch failed: {fetched.get('error')}"}

    ctype = fetched.get("ctype", "") or ""
    html = fetched.get("text", "") or ""
    if "text/html" not in ctype and len(html) < 200:
        # try action=render as another fallback
        if progress:
            progress("Content type not HTML or too small; trying action=render fallback ...")
        sep = "&" if "?" in url else "?"
        render_url = url + sep + "action=render"
        fetched2 = _fetch_html(render_url, headers=headers)
        if fetched2.get("ok"):
            html2 = fetched2.get("text", "") or ""
            if len(html2) > len(html):
                html = html2
                ctype = fetched2.get("ctype", ctype)

    if not html or len(html) < 80:
        if progress:
            progress("No meaningful HTML body fetched.")
        return {"ok": False, "error": "no-text"}

    # extract cleaned text
    if progress:
        progress("Extracting text from HTML ...")
    text = _clean_html_text(html)
    if not text or len(text) < 50:
        if progress:
            progress("Extracted text too short; trying render fallback (if not already tried)...")
        sep = "&" if "?" in url else "?"
        render_url = url + sep + "action=render"
        fetched3 = _fetch_html(render_url, headers=headers)
        if fetched3.get("ok"):
            text = _clean_html_text(fetched3.get("text", "") or "")
    if not text or len(text) < 50:
        if progress:
            progress("No meaningful text extracted from page.")
        return {"ok": False, "error": "no-text"}

    if progress:
        progress("Chunking text ...")
    chunks = _chunk_text(text)
    added = 0
    ids = []
    for i, ch in enumerate(chunks[:max_chunks]):
        if progress and (i % 5 == 0):
            progress(f"Storing chunk {i+1}/{min(len(chunks), max_chunks)} ...")
        try:
            # memory API: add_chunk(text, title=..., source=..., url=..., ts=...)
            item_id = memory.add_chunk(ch, title=(parsed.path or ""), source=parsed.netloc, url=url, ts=int(time.time()))
        except Exception:
            # try alternative name if memory uses different method
            try:
                item_id = memory.add(ch, title=(parsed.path or ""), source=parsed.netloc, url=url, ts=int(time.time()))
            except Exception:
                item_id = None
        if item_id:
            added += 1
            ids.append(item_id)

    if progress:
        progress(f"Done. {added} new chunk(s) stored (out of {len(chunks)} candidates).")

    title = None
    try:
        # try to pick a title from HTML <title>
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
    except Exception:
        title = None

    return {"ok": True, "added": added, "total_candidates": len(chunks), "ids": ids, "title": title}
