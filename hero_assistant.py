#!/usr/bin/env python3
"""
hero_assistant.py

Model-first assistant for HeroX:
- conversational chat via external model (Gemini/OpenAI adapter)
- persistent memory (user-taught facts/strategies)
- sequential task queue (assistant executes tasks in order)
- trade outcome learning + notifications
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

from openai_api_placeholder import get_last_llm_status, maybe_generate_with_openai


def _now_ts() -> int:
    return int(time.time())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _compact(obj: Any, max_depth: int = 4, max_list: int = 40, max_str: int = 400, _depth: int = 0) -> Any:
    if _depth >= max_depth:
        return "[max_depth]"
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if str(k) in ("__raw", "raw", "debug_dump", "big_payload"):
                continue
            out[str(k)] = _compact(v, max_depth=max_depth, max_list=max_list, max_str=max_str, _depth=_depth + 1)
        return out
    if isinstance(obj, list):
        arr = obj[:max_list]
        out = [_compact(x, max_depth=max_depth, max_list=max_list, max_str=max_str, _depth=_depth + 1) for x in arr]
        if len(obj) > max_list:
            out.append(f"...({len(obj) - max_list} more)")
        return out
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else (obj[:max_str] + "...[truncated]")
    return obj


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "").strip()
    if not s:
        return None

    # Common model format: fenced JSON block.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(s[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _extract_json_data(text: str) -> Optional[Any]:
    s = str(text or "").strip()
    if not s:
        return None

    m = re.search(r"```(?:json)?\s*(.+?)\s*```", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        try:
            obj, _ = dec.raw_decode(s[i:])
            return obj
        except Exception:
            continue
    return None


def _norm_key(v: Any) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9\s\-_.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _trim_text(v: Any, max_len: int = 1200) -> str:
    s = str(v or "").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len].rstrip() + "..."


class HeroAssistantEngine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.base_dir = os.environ.get("HERO_ASSISTANT_DIR", os.path.expanduser("~/.hero_logs/assistant"))
        os.makedirs(self.base_dir, exist_ok=True)

        self.state_file = os.path.join(self.base_dir, "state.json")
        self.events_log = os.path.join(self.base_dir, "events.jsonl")
        self.outcomes_log = os.path.join(self.base_dir, "outcomes.jsonl")
        self.notifications_log = os.path.join(self.base_dir, "notifications.jsonl")
        self.learned_log = os.path.join(self.base_dir, "learned.jsonl")
        self.knowledge_log = os.path.join(self.base_dir, "knowledge.jsonl")
        self.news_log = os.path.join(self.base_dir, "news.jsonl")

        self.memory_capacity = max(100, int(os.environ.get("HERO_ASSISTANT_MEMORY_CAP", "5000")))
        self.knowledge_capacity = max(120, int(os.environ.get("HERO_ASSISTANT_KNOWLEDGE_CAP", "1200")))
        self.news_capacity = max(60, int(os.environ.get("HERO_ASSISTANT_NEWS_CAP", "300")))
        self.news_ttl_sec = max(600, int(os.environ.get("HERO_ASSISTANT_NEWS_TTL_SEC", "172800")))
        self.research_interval_sec = max(60, int(os.environ.get("HERO_ASSISTANT_RESEARCH_INTERVAL_SEC", "120")))
        self.auto_research_enabled = str(os.environ.get("HERO_ASSISTANT_AUTO_RESEARCH", "1")).strip().lower() in ("1", "true", "yes", "on")
        self.research_transient_backoff_sec = max(self.research_interval_sec, int(os.environ.get("HERO_ASSISTANT_RESEARCH_TRANSIENT_BACKOFF_SEC", "300")))
        self.research_auth_backoff_sec = max(self.research_interval_sec, int(os.environ.get("HERO_ASSISTANT_RESEARCH_AUTH_BACKOFF_SEC", "21600")))
        self.research_error_notify_cooldown_sec = max(60, int(os.environ.get("HERO_ASSISTANT_RESEARCH_ERROR_NOTIFY_COOLDOWN_SEC", "1800")))
        self.auto_topics: List[str] = [
            "forex market structure and trend continuation setups",
            "deriv higher/lower and rise/fall strategy updates",
            "technical indicators: RSI, MACD, EMA, Bollinger Bands, ATR, ADX, Stochastic",
            "chart patterns and price action confirmation rules",
            "risk management and trade sizing in volatile markets",
            "stock market and crypto market macro drivers relevant to short-term trading",
            "major financial/forex/crypto news that may impact volatility",
        ]
        extra_topics = [x.strip() for x in str(os.environ.get("HERO_ASSISTANT_RESEARCH_TOPICS", "")).split("|") if x.strip()]
        if extra_topics:
            self.auto_topics = extra_topics

        self.memory_items: List[Dict[str, Any]] = []
        self.learned_items: List[Dict[str, Any]] = []
        self.knowledge_items: List[Dict[str, Any]] = []
        self.news_items: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []

        self.pending_predictions: Dict[str, Dict[str, Any]] = {}
        self.latest_market_state: Dict[str, Dict[str, Any]] = {}
        self.market_tick_tail: Dict[str, deque] = {}
        self.recent_outcomes: deque = deque(maxlen=1200)
        self.notifications: deque = deque(maxlen=600)
        self.chat_history: deque = deque(maxlen=240)
        self.last_tick_ts = 0
        self.last_analysis_event_ts = 0

        self.total_tick_events = 0
        self.total_analysis_events = 0
        self.total_research_cycles = 0
        self.last_research_ts = 0
        self.last_research_status = "idle"
        self.last_research_error = ""
        self.research_pause_until_ts = 0
        self._notify_recent: Dict[str, int] = {}

        self._load_state()
        self._load_logs()

        self._task_thread = threading.Thread(target=self._task_worker_loop, daemon=True, name="hero_assistant_task_worker")
        self._task_thread.start()
        if self.auto_research_enabled:
            self._research_thread = threading.Thread(target=self._auto_research_loop, daemon=True, name="hero_assistant_auto_research")
            self._research_thread.start()

    # ---------------- persistence ----------------
    def _append_jsonl(self, path: str, obj: Dict[str, Any]) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=True, default=str) + "\n")
        except Exception:
            pass

    def _save_state(self) -> None:
        payload = {
            "saved_at": _now_iso(),
            "memory_capacity": self.memory_capacity,
            "memory_items": self.memory_items[-self.memory_capacity :],
            "learned_items": self.learned_items[-500:],
            "knowledge_capacity": self.knowledge_capacity,
            "news_capacity": self.news_capacity,
            "knowledge_items": self.knowledge_items[-self.knowledge_capacity :],
            "news_items": self.news_items[-self.news_capacity :],
            "tasks": self.tasks[-500:],
            "recent_outcomes": list(self.recent_outcomes)[-1000:],
            "chat_history": list(self.chat_history)[-240:],
            "last_tick_ts": int(self.last_tick_ts or 0),
            "last_analysis_event_ts": int(self.last_analysis_event_ts or 0),
            "last_research_ts": int(self.last_research_ts or 0),
            "last_research_status": str(self.last_research_status or "idle"),
            "last_research_error": str(self.last_research_error or ""),
            "research_pause_until_ts": int(self.research_pause_until_ts or 0),
            "total_research_cycles": int(self.total_research_cycles or 0),
            "total_tick_events": self.total_tick_events,
            "total_analysis_events": self.total_analysis_events,
        }
        tmp = self.state_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True)
            os.replace(tmp, self.state_file)
        except Exception:
            pass

    def _load_state(self) -> None:
        try:
            if not os.path.exists(self.state_file):
                return
            with open(self.state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return
            self.memory_capacity = int(payload.get("memory_capacity") or self.memory_capacity)
            self.knowledge_capacity = int(payload.get("knowledge_capacity") or self.knowledge_capacity)
            self.news_capacity = int(payload.get("news_capacity") or self.news_capacity)
            self.memory_items = list(payload.get("memory_items") or [])[-self.memory_capacity :]
            self.learned_items = list(payload.get("learned_items") or [])[-500:]
            self.knowledge_items = list(payload.get("knowledge_items") or [])[-self.knowledge_capacity :]
            self.news_items = list(payload.get("news_items") or [])[-self.news_capacity :]
            self.tasks = list(payload.get("tasks") or [])[-500:]
            for row in list(payload.get("recent_outcomes") or [])[-1000:]:
                if isinstance(row, dict):
                    self.recent_outcomes.append(row)
            for row in list(payload.get("chat_history") or [])[-240:]:
                if isinstance(row, dict):
                    self.chat_history.append(row)
            self.last_tick_ts = int(payload.get("last_tick_ts") or 0)
            self.last_analysis_event_ts = int(payload.get("last_analysis_event_ts") or 0)
            self.last_research_ts = int(payload.get("last_research_ts") or 0)
            self.last_research_status = str(payload.get("last_research_status") or "idle")
            self.last_research_error = str(payload.get("last_research_error") or "")
            self.research_pause_until_ts = int(payload.get("research_pause_until_ts") or 0)
            self.total_research_cycles = int(payload.get("total_research_cycles") or 0)
            self.total_tick_events = int(payload.get("total_tick_events") or 0)
            self.total_analysis_events = int(payload.get("total_analysis_events") or 0)
        except Exception:
            pass

    def _load_logs(self) -> None:
        # notifications
        try:
            if os.path.exists(self.notifications_log):
                with open(self.notifications_log, "r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            row = json.loads(ln)
                            if isinstance(row, dict):
                                self.notifications.append(row)
                        except Exception:
                            continue
        except Exception:
            pass

        # outcomes rebuild if empty
        if len(self.recent_outcomes) > 0:
            pass
        else:
            try:
                if os.path.exists(self.outcomes_log):
                    with open(self.outcomes_log, "r", encoding="utf-8") as f:
                        for ln in f:
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                row = json.loads(ln)
                                if isinstance(row, dict):
                                    self.recent_outcomes.append(row)
                            except Exception:
                                continue
            except Exception:
                pass

        if len(self.learned_items) == 0:
            try:
                if os.path.exists(self.learned_log):
                    with open(self.learned_log, "r", encoding="utf-8") as f:
                        for ln in f:
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                row = json.loads(ln)
                                if isinstance(row, dict):
                                    self.learned_items.append(row)
                            except Exception:
                                continue
                self.learned_items = self.learned_items[-500:]
            except Exception:
                pass

        if len(self.knowledge_items) == 0:
            try:
                if os.path.exists(self.knowledge_log):
                    with open(self.knowledge_log, "r", encoding="utf-8") as f:
                        for ln in f:
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                row = json.loads(ln)
                                if isinstance(row, dict):
                                    self.knowledge_items.append(row)
                            except Exception:
                                continue
                self.knowledge_items = self.knowledge_items[-self.knowledge_capacity :]
            except Exception:
                pass

        if len(self.news_items) == 0:
            try:
                if os.path.exists(self.news_log):
                    with open(self.news_log, "r", encoding="utf-8") as f:
                        for ln in f:
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                row = json.loads(ln)
                                if isinstance(row, dict):
                                    self.news_items.append(row)
                            except Exception:
                                continue
                self.news_items = self.news_items[-self.news_capacity :]
            except Exception:
                pass

    # ---------------- helpers ----------------
    def _market_of(self, payload: Dict[str, Any]) -> str:
        return str(payload.get("market") or payload.get("symbol") or payload.get("market_code") or "").upper()

    def _prediction_id(self, payload: Dict[str, Any]) -> str:
        pid = payload.get("prediction_id") or payload.get("pred_id") or payload.get("id")
        if pid:
            return str(pid)
        return f"pred_{_now_ts()}_{self._market_of(payload) or 'UNKNOWN'}"

    def _extract_indicators(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if isinstance(payload.get("indicators"), dict):
            return payload.get("indicators")
        reason = payload.get("reason")
        if isinstance(reason, dict) and isinstance(reason.get("indicators"), dict):
            return reason.get("indicators")
        return None

    def _extract_direction(self, payload: Dict[str, Any]) -> str:
        for key in ("direction", "signal", "trade_type", "mode"):
            v = payload.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
        ctype = str(payload.get("contract_type") or "").upper()
        if ctype == "CALL":
            return "higher"
        if ctype == "PUT":
            return "lower"
        return "unknown"

    def _extract_outcome(self, payload: Dict[str, Any]) -> str:
        raw = payload.get("result") or payload.get("outcome") or payload.get("status")
        if isinstance(raw, str):
            u = raw.strip().upper()
            if u in ("WIN", "LOSS"):
                return u
        profit = _safe_float(payload.get("profit"))
        if profit is not None:
            return "WIN" if profit > 0 else "LOSS"
        return "UNKNOWN"

    def _is_auth_error_text(self, message: str) -> bool:
        low = str(message or "").lower()
        return (
            ("http 401" in low)
            or ("http 403" in low)
            or ("permission_denied" in low)
            or ("unauthenticated" in low)
            or ("api key was reported as leaked" in low)
            or ("invalid api key" in low)
            or ("auth/permission error" in low)
            or ("update gemini_api_key" in low)
            or (("api key" in low or "gemini_api_key" in low) and ("invalid" in low or "leaked" in low))
            or (("api key" in low) and ("forbidden" in low))
        )

    def _normalize_research_error(self, message: str) -> str:
        txt = _trim_text(message, max_len=320)
        if self._is_auth_error_text(txt):
            return "LLM auth/permission error. Update GEMINI_API_KEY (old key appears invalid or leaked)."
        return txt

    def _notify(
        self,
        level: str,
        message: str,
        market: Optional[str] = None,
        outcome: Optional[str] = None,
        prediction_id: Optional[str] = None,
        dedupe_window_sec: int = 0,
    ) -> None:
        ts_now = _now_ts()
        dedupe_for = max(0, int(dedupe_window_sec or 0))
        sig = f"{str(level)}|{str(market)}|{str(outcome)}|{str(prediction_id)}|{str(message)}"
        if dedupe_for > 0:
            with self._lock:
                last = int(self._notify_recent.get(sig) or 0)
                if ts_now - last < dedupe_for:
                    return
                self._notify_recent[sig] = ts_now
                if len(self._notify_recent) > 2000:
                    cutoff = ts_now - 86400
                    self._notify_recent = {k: v for k, v in self._notify_recent.items() if int(v) >= cutoff}
        row = {
            "ts": ts_now,
            "level": level,
            "market": market,
            "outcome": outcome,
            "prediction_id": prediction_id,
            "message": message,
        }
        with self._lock:
            self.notifications.append(row)
            self._append_jsonl(self.notifications_log, row)

    def _extract_task_topic(self, task: Dict[str, Any]) -> str:
        topic = str(task.get("topic") or "").strip()
        if topic:
            return topic
        text = str(task.get("text") or "").strip()
        prefixes = (
            "Research and learn deeply about:",
            "Research and learn about:",
            "Learn about:",
            "Research:",
        )
        for p in prefixes:
            if text.lower().startswith(p.lower()):
                return text[len(p) :].strip() or text
        return text

    def _normalize_learning_record(self, raw: Dict[str, Any], topic: str, task_id: str, source_text: str) -> Dict[str, Any]:
        title = _trim_text(raw.get("title") or topic or "Untitled learning", max_len=140)
        summary = _trim_text(raw.get("summary") or raw.get("what_i_learned") or source_text, max_len=1200)
        takeaway = _trim_text(raw.get("takeaway") or raw.get("key_takeaway") or summary, max_len=360)

        subtopics_in = raw.get("subtopics")
        subtopics: List[Dict[str, Any]] = []
        if isinstance(subtopics_in, list):
            for row in subtopics_in[:10]:
                if not isinstance(row, dict):
                    continue
                st_title = _trim_text(row.get("title") or row.get("name") or "Subtopic", max_len=120)
                learned = _trim_text(row.get("learned") or row.get("details") or row.get("content") or "", max_len=700)
                st_takeaway = _trim_text(row.get("takeaway") or "", max_len=220)
                if not learned and not st_takeaway:
                    continue
                subtopics.append({"title": st_title, "learned": learned, "takeaway": st_takeaway})

        if not subtopics:
            lines = [ln.strip("-* \t") for ln in source_text.splitlines() if ln.strip()]
            head = lines[:9]
            for i in range(0, len(head), 2):
                t = _trim_text(head[i], max_len=120)
                learned = _trim_text(head[i + 1] if i + 1 < len(head) else head[i], max_len=380)
                subtopics.append({"title": t or f"Point {i+1}", "learned": learned, "takeaway": ""})
            if not subtopics:
                subtopics = [{"title": "Overview", "learned": _trim_text(source_text, max_len=650), "takeaway": takeaway}]

        return {
            "id": f"learn_{_now_ts()}_{len(self.learned_items) + 1}",
            "task_id": task_id,
            "topic": _trim_text(topic, max_len=220),
            "title": title,
            "summary": summary,
            "takeaway": takeaway,
            "subtopics": subtopics[:8],
            "created_ts": _now_ts(),
        }

    def _build_learning_record(self, task: Dict[str, Any], task_result: str) -> Dict[str, Any]:
        topic = self._extract_task_topic(task)
        task_id = str(task.get("id") or "")
        result_text = _trim_text(task_result, max_len=5000)

        prompt = (
            "Convert the learning notes below into strict JSON.\n"
            "Return JSON only with this schema:\n"
            "{\n"
            '  "title": "short title",\n'
            '  "summary": "concise summary",\n'
            '  "takeaway": "single key takeaway",\n'
            '  "subtopics": [\n'
            '    {"title":"...", "learned":"...", "takeaway":"..."}\n'
            "  ]\n"
            "}\n"
            "Rules: 3-6 subtopics, practical, plain language, no markdown fences."
        )
        context = {
            "allow_web_search": False,
            "task_id": task_id,
            "task_topic": topic,
            "task_text": task.get("text"),
            "task_result": result_text,
        }
        raw = maybe_generate_with_openai(prompt, context)
        parsed = _extract_json_object(raw) if isinstance(raw, str) else None
        if isinstance(parsed, dict):
            return self._normalize_learning_record(parsed, topic=topic, task_id=task_id, source_text=result_text)
        return self._normalize_learning_record({}, topic=topic, task_id=task_id, source_text=result_text)

    def _add_learned_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        rec = dict(item or {})
        if not rec:
            rec = {
                "id": f"learn_{_now_ts()}_{len(self.learned_items) + 1}",
                "title": "Untitled learning",
                "summary": "",
                "takeaway": "",
                "subtopics": [],
                "created_ts": _now_ts(),
            }
        self.learned_items.append(rec)
        self.learned_items = self.learned_items[-500:]
        self._append_jsonl(self.learned_log, rec)
        self._save_state()
        return rec

    def _upsert_knowledge_item(self, row: Dict[str, Any], topic: str) -> bool:
        title = _trim_text(row.get("title") or row.get("topic") or topic, max_len=180)
        summary = _trim_text(row.get("summary") or row.get("details") or row.get("learned") or "", max_len=1400)
        if not title and not summary:
            return False
        key = _norm_key(title or f"{topic} {summary[:80]}")
        if not key:
            return False
        importance = int(max(1, min(100, round(_safe_float(row.get("importance")) or 50))))
        category = _trim_text(row.get("category") or row.get("domain") or "trading", max_len=80)
        takeaway = _trim_text(row.get("takeaway") or row.get("why_it_matters") or "", max_len=340)
        tags_in = row.get("tags")
        tags: List[str] = []
        if isinstance(tags_in, list):
            tags = [_trim_text(x, max_len=40) for x in tags_in if _trim_text(x, max_len=40)]
        if topic and topic not in tags:
            tags.append(_trim_text(topic, max_len=80))
        now = _now_ts()
        for item in self.knowledge_items:
            if str(item.get("key")) == key:
                if summary:
                    item["summary"] = summary
                if takeaway:
                    item["takeaway"] = takeaway
                item["importance"] = max(int(item.get("importance") or 0), importance)
                item["category"] = category or str(item.get("category") or "trading")
                item["tags"] = list(dict.fromkeys((item.get("tags") or []) + tags))[:12]
                item["updated_ts"] = now
                item["hits"] = int(item.get("hits") or 0) + 1
                return False

        rec = {
            "id": f"k_{now}_{len(self.knowledge_items) + 1}",
            "key": key,
            "title": title or _trim_text(topic, max_len=120) or "Trading knowledge",
            "summary": summary,
            "takeaway": takeaway,
            "category": category,
            "importance": importance,
            "tags": tags[:12],
            "created_ts": now,
            "updated_ts": now,
            "hits": 1,
        }
        self.knowledge_items.append(rec)
        self._append_jsonl(self.knowledge_log, rec)
        return True

    def _upsert_news_item(self, row: Dict[str, Any], topic: str) -> bool:
        title = _trim_text(row.get("title") or row.get("headline") or "", max_len=200)
        summary = _trim_text(row.get("summary") or row.get("details") or row.get("why_it_matters") or "", max_len=1200)
        if not title:
            return False
        key = _norm_key(title)
        if not key:
            return False
        importance = int(max(1, min(100, round(_safe_float(row.get("importance")) or 50))))
        market = _trim_text(row.get("market") or row.get("asset") or topic or "forex", max_len=80)
        source = _trim_text(row.get("source") or row.get("publisher") or "", max_len=120)
        now = _now_ts()
        for item in self.news_items:
            if str(item.get("key")) == key:
                item["summary"] = summary or str(item.get("summary") or "")
                item["importance"] = max(int(item.get("importance") or 0), importance)
                item["market"] = market or str(item.get("market") or "")
                item["source"] = source or str(item.get("source") or "")
                item["updated_ts"] = now
                item["hits"] = int(item.get("hits") or 0) + 1
                return False

        rec = {
            "id": f"n_{now}_{len(self.news_items) + 1}",
            "key": key,
            "title": title,
            "summary": summary,
            "importance": importance,
            "market": market,
            "source": source,
            "created_ts": now,
            "updated_ts": now,
            "hits": 1,
        }
        self.news_items.append(rec)
        self._append_jsonl(self.news_log, rec)
        return True

    def _prune_knowledge_and_news(self) -> None:
        now = _now_ts()
        # Keep very high-importance headlines even if older; prune stale low-importance news.
        kept_news: List[Dict[str, Any]] = []
        for n in self.news_items:
            upd = int(n.get("updated_ts") or n.get("created_ts") or now)
            age = max(0, now - upd)
            imp = int(n.get("importance") or 0)
            if age <= self.news_ttl_sec or imp >= 80:
                kept_news.append(n)
        kept_news.sort(key=lambda x: (int(x.get("importance") or 0), int(x.get("updated_ts") or 0)), reverse=True)
        self.news_items = kept_news[: self.news_capacity]

        # Knowledge: keep highest weighted by importance + freshness + hits.
        def _k_score(k: Dict[str, Any]) -> float:
            imp = float(k.get("importance") or 0.0)
            upd = float(k.get("updated_ts") or k.get("created_ts") or now)
            age_h = max(0.0, (now - upd) / 3600.0)
            freshness = max(0.0, 72.0 - min(age_h, 72.0)) / 72.0
            hits = float(k.get("hits") or 1.0)
            return imp * 1.0 + freshness * 12.0 + min(hits, 10.0) * 1.5

        self.knowledge_items.sort(key=_k_score, reverse=True)
        self.knowledge_items = self.knowledge_items[: self.knowledge_capacity]

    def _is_strategy_like(self, title: str, category: str, tags: Optional[List[str]] = None) -> bool:
        txt = " ".join(
            [
                str(title or ""),
                str(category or ""),
                " ".join([str(x) for x in (tags or [])]),
            ]
        ).lower()
        keys = ("strategy", "setup", "pattern", "entry", "breakout", "reversal", "indicator", "trend", "price action")
        return any(k in txt for k in keys)

    def _strategy_count(self) -> int:
        n = 0
        for k in self.knowledge_items:
            if self._is_strategy_like(str(k.get("title") or ""), str(k.get("category") or ""), list(k.get("tags") or [])):
                n += 1
        return n

    def _coerce_research_payload(self, raw_text: str, topic: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"knowledge": [], "news": []}
        raw = str(raw_text or "").strip()
        if not raw:
            return out

        first = _extract_json_data(raw)
        if isinstance(first, dict):
            k = first.get("knowledge")
            n = first.get("news")
            out["knowledge"] = k if isinstance(k, list) else []
            out["news"] = n if isinstance(n, list) else []
            if out["knowledge"] or out["news"]:
                return out

        # Second pass: ask model to normalize arbitrary output to strict JSON.
        normalize_prompt = (
            "Convert the following content to strict JSON only.\n"
            "Schema:\n"
            "{\n"
            '  "knowledge":[{"title":"...","summary":"...","category":"...","importance":70,"tags":["..."],"takeaway":"..."}],\n'
            '  "news":[{"title":"...","summary":"...","market":"...","source":"...","importance":75}]\n'
            "}\n"
            "Rules: no markdown fences; if unknown use empty arrays."
        )
        normalize_ctx = {
            "allow_web_search": False,
            "topic": topic,
            "raw_content": _trim_text(raw, max_len=9000),
        }
        norm = maybe_generate_with_openai(normalize_prompt, normalize_ctx)
        parsed = _extract_json_data(norm) if isinstance(norm, str) else None
        if isinstance(parsed, dict):
            k = parsed.get("knowledge")
            n = parsed.get("news")
            out["knowledge"] = k if isinstance(k, list) else []
            out["news"] = n if isinstance(n, list) else []
            if out["knowledge"] or out["news"]:
                return out

        # Last-resort heuristic: keep useful lines as knowledge so cycle never fully fails.
        lines = [ln.strip(" -*\t") for ln in raw.splitlines() if ln.strip()]
        if lines:
            items: List[Dict[str, Any]] = []
            for ln in lines[:8]:
                items.append(
                    {
                        "title": _trim_text(ln[:120], max_len=120),
                        "summary": _trim_text(ln, max_len=800),
                        "category": "trading_research",
                        "importance": 55,
                        "tags": [topic],
                        "takeaway": _trim_text(ln, max_len=220),
                    }
                )
            out["knowledge"] = items
        return out

    def _auto_research_once(self) -> None:
        topics = list(self.auto_topics or [])
        if not topics:
            return
        topic = topics[self.total_research_cycles % len(topics)]
        prompt = (
            "Run a concise web research sweep and return strict JSON only.\n"
            "Focus on practical trading education and actionable updates.\n"
            "JSON schema:\n"
            "{\n"
            '  "knowledge": [\n'
            '    {"title":"...", "summary":"...", "category":"...", "importance":70, "tags":["..."], "takeaway":"..."}\n'
            "  ],\n"
            '  "news": [\n'
            '    {"title":"...", "summary":"...", "market":"forex/deriv/crypto/stocks", "source":"...", "importance":75}\n'
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- 4 to 10 knowledge items\n"
            "- 3 to 8 news items\n"
            "- prioritize forex + deriv, but include stock/crypto macro only when relevant\n"
            "- no markdown fences, no prose outside JSON."
        )
        context = {
            "allow_web_search": True,
            "mode": "auto_research",
            "research_topic": topic,
            "latest_markets": _compact(self.latest_market_state),
            "recent_outcomes": _compact(list(self.recent_outcomes)[-120:]),
            "known_knowledge_titles": [x.get("title") for x in self.knowledge_items[-300:]],
            "known_news_titles": [x.get("title") for x in self.news_items[-200:]],
        }
        out = maybe_generate_with_openai(prompt, context)
        if not isinstance(out, str) or not out.strip():
            llm = get_last_llm_status()
            err_txt = self._normalize_research_error(str(llm.get("error") or "model_unavailable"))
            raise RuntimeError(err_txt)
        payload = self._coerce_research_payload(out, topic=topic)
        if not isinstance(payload, dict):
            payload = {"knowledge": [], "news": []}

        knowledge_rows = payload.get("knowledge")
        news_rows = payload.get("news")
        if not isinstance(knowledge_rows, list):
            knowledge_rows = []
        if not isinstance(news_rows, list):
            news_rows = []

        new_k = 0
        new_n = 0
        strategy_added = 0
        major_news_added: List[Dict[str, Any]] = []
        with self._lock:
            for row in knowledge_rows:
                if isinstance(row, dict):
                    added = self._upsert_knowledge_item(row, topic=topic)
                    if added:
                        new_k += 1
                        if self._is_strategy_like(str(row.get("title") or ""), str(row.get("category") or ""), list(row.get("tags") or [])):
                            strategy_added += 1
            for row in news_rows:
                if isinstance(row, dict):
                    added = self._upsert_news_item(row, topic=topic)
                    if added:
                        new_n += 1
                        imp = int(max(0, min(100, round(_safe_float(row.get("importance")) or 0))))
                        if imp >= 85:
                            major_news_added.append(row)
            self._prune_knowledge_and_news()
            self.total_research_cycles += 1
            self.last_research_ts = _now_ts()
            self.last_research_status = "ok"
            self.last_research_error = ""
            self._save_state()

        if new_k > 0 or new_n > 0:
            self._notify("info", f"Auto research updated: +{new_k} knowledge, +{new_n} news", market=topic)
        if strategy_added > 0:
            self._notify("info", f"Strategies learned this cycle: {strategy_added}", market=topic)
        total_strategies = self._strategy_count()
        self._notify("info", f"Strategy library size: {total_strategies}", market=topic)
        for row in major_news_added[:2]:
            self._notify("info", f"Major market news: {_trim_text(row.get('title') or '', max_len=180)}", market=str(row.get("market") or topic))

    def _auto_research_loop(self) -> None:
        while True:
            now = _now_ts()
            if int(self.research_pause_until_ts or 0) > now:
                time.sleep(min(30, int(self.research_pause_until_ts - now)))
                continue
            try:
                self._auto_research_once()
                with self._lock:
                    self.research_pause_until_ts = 0
            except Exception as e:
                err_txt = self._normalize_research_error(str(e))
                is_auth = self._is_auth_error_text(str(e)) or self._is_auth_error_text(err_txt)
                with self._lock:
                    self.last_research_ts = _now_ts()
                    self.last_research_status = "blocked_auth" if is_auth else "error"
                    self.last_research_error = _trim_text(err_txt, max_len=240)
                    if is_auth:
                        self.research_pause_until_ts = max(
                            int(self.research_pause_until_ts or 0),
                            _now_ts() + self.research_auth_backoff_sec,
                        )
                    self._save_state()
                if is_auth:
                    pause_until = datetime.fromtimestamp(int(self.research_pause_until_ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
                    self._notify(
                        "warn",
                        f"Auto research paused: {self.last_research_error} Next retry after {pause_until}.",
                        dedupe_window_sec=self.research_auth_backoff_sec,
                    )
                    time.sleep(min(self.research_interval_sec, 30))
                    continue
                self._notify(
                    "warn",
                    f"Auto research error: {self.last_research_error}",
                    dedupe_window_sec=self.research_error_notify_cooldown_sec,
                )
                time.sleep(self.research_transient_backoff_sec)
                continue
            time.sleep(self.research_interval_sec)

    # ---------------- ingestion ----------------
    def ingest_tick(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        with self._lock:
            self.total_tick_events += 1
            self.last_tick_ts = _now_ts()
            market = self._market_of(payload)
            if not market or market == "ANALYSIS":
                return
            if market not in self.market_tick_tail:
                self.market_tick_tail[market] = deque(maxlen=40)
            d = payload.get("last_decimal")
            try:
                if d is not None:
                    self.market_tick_tail[market].append(int(d))
            except Exception:
                pass
            self.latest_market_state[market] = {
                "ts": _now_ts(),
                "price": payload.get("price"),
                "last_decimal": payload.get("last_decimal"),
                "reason": payload.get("reason"),
                "indicators": self._extract_indicators(payload),
            }

    def ingest_analysis(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        with self._lock:
            self.total_analysis_events += 1
            self.last_analysis_event_ts = _now_ts()
            event = str(payload.get("analysis_event") or payload.get("event") or "").lower()
            market = self._market_of(payload)
            pid = self._prediction_id(payload)
            direction = self._extract_direction(payload)
            indicators = self._extract_indicators(payload)
            confidence = _safe_float(payload.get("confidence"))

            self._append_jsonl(
                self.events_log,
                {
                    "ts": _now_ts(),
                    "event": event,
                    "market": market,
                    "prediction_id": pid,
                    "payload": _compact(payload),
                },
            )

            if market:
                prev = self.latest_market_state.get(market) or {}
                self.latest_market_state[market] = {
                    "ts": _now_ts(),
                    "price": payload.get("price", prev.get("price")),
                    "last_decimal": payload.get("last_decimal", prev.get("last_decimal")),
                    "reason": payload.get("reason", prev.get("reason")),
                    "indicators": indicators if indicators is not None else prev.get("indicators"),
                }

            if event in ("trade_posted", "prediction_posted", "signal"):
                self.pending_predictions[pid] = {
                    "ts": _now_ts(),
                    "market": market,
                    "direction": direction,
                    "confidence": confidence,
                    "reason": payload.get("reason"),
                    "indicators": indicators,
                    "raw": _compact(payload),
                }
                return

            # settled/final events
            settled = (
                event in ("prediction_result", "trade_result", "settled", "contract_settled")
                or payload.get("settled") is True
                or (payload.get("result") in ("WIN", "LOSS"))
                or (payload.get("outcome") in ("WIN", "LOSS"))
            )
            if not settled:
                return

            outcome = self._extract_outcome(payload)
            if outcome == "UNKNOWN":
                return

            pending = self.pending_predictions.pop(pid, {})
            rec = {
                "ts": _now_ts(),
                "prediction_id": pid,
                "market": market or str(pending.get("market") or "").upper(),
                "direction": direction if direction != "unknown" else str(pending.get("direction") or "unknown"),
                "confidence": confidence if confidence is not None else _safe_float(pending.get("confidence")),
                "outcome": outcome,
                "profit": payload.get("profit"),
                "reason": payload.get("reason") if payload.get("reason") is not None else pending.get("reason"),
                "indicators": indicators if indicators is not None else pending.get("indicators"),
            }
            self.recent_outcomes.append(rec)
            self._append_jsonl(self.outcomes_log, rec)
            self._save_state()

            note = self._build_outcome_explanation(rec)
            self._notify("info" if outcome == "WIN" else "warn", note, market=rec.get("market"), outcome=outcome, prediction_id=pid)

    # ---------------- tasks ----------------
    def _queued_tasks(self) -> List[Dict[str, Any]]:
        return [t for t in self.tasks if str(t.get("status")) == "queued"]

    def _task_worker_loop(self) -> None:
        while True:
            try:
                with self._lock:
                    queued = self._queued_tasks()
                    task = queued[0] if queued else None
                    if task:
                        task["status"] = "running"
                        task["started_ts"] = _now_ts()
                        self._notify("info", f"Task started: {task.get('text')}")
                        self._save_state()
                if not task:
                    time.sleep(1.0)
                    continue

                prompt = (
                    f"Execute this task and report completion clearly.\nTask: {task.get('text')}\n"
                    "Use web research if needed. Return concise findings and practical steps."
                )
                context = {
                    "allow_web_search": True,
                    "task": task,
                    "memory_items": self.memory_items[-150:],
                    "latest_market_state": _compact(self.latest_market_state),
                    "recent_outcomes": _compact(list(self.recent_outcomes)[-100:]),
                }
                result = maybe_generate_with_openai(prompt, context)
                llm = get_last_llm_status()
                learning_record: Optional[Dict[str, Any]] = None
                if isinstance(result, str) and result.strip():
                    learning_record = self._build_learning_record(task, result.strip())

                with self._lock:
                    if isinstance(result, str) and result.strip():
                        learned = self._add_memory(
                            f"Task learning: {task.get('text')}\nFindings:\n{result.strip()}",
                            source="task",
                        )
                        learned_item = self._add_learned_item(learning_record or {})
                        task["status"] = "done"
                        task["result"] = result.strip()
                        task["memory_id"] = learned.get("id")
                        task["learned_id"] = learned_item.get("id")
                        task["done_ts"] = _now_ts()
                        self._notify(
                            "info",
                            f"Learning complete: {learned_item.get('title')}",
                            market=str(task.get("topic") or self._extract_task_topic(task) or ""),
                        )
                    else:
                        task["status"] = "failed"
                        task["error"] = llm.get("error") or "model_unavailable"
                        task["done_ts"] = _now_ts()
                        self._notify("warn", f"Task failed: {task.get('text')} ({task.get('error')})")
                    self._save_state()
            except Exception:
                time.sleep(1.0)

    def _add_task(self, text: str, kind: str = "task", topic: Optional[str] = None) -> Dict[str, Any]:
        item = {
            "id": f"task_{_now_ts()}_{len(self.tasks) + 1}",
            "text": text.strip(),
            "kind": (kind or "task").strip().lower(),
            "topic": (topic or "").strip(),
            "status": "queued",
            "created_ts": _now_ts(),
        }
        self.tasks.append(item)
        self.tasks = self.tasks[-500:]
        self._save_state()
        note = f"Learning queued: {(topic or text).strip()}" if item["kind"] == "learn" else f"Task queued: {text.strip()}"
        self._notify("info", note, market=(topic or None))
        return item

    # ---------------- memory ----------------
    def _add_memory(self, text: str, source: str = "chat") -> Dict[str, Any]:
        item = {
            "id": f"mem_{_now_ts()}_{len(self.memory_items) + 1}",
            "text": text.strip(),
            "source": source,
            "created_ts": _now_ts(),
        }
        self.memory_items.append(item)
        if len(self.memory_items) > self.memory_capacity:
            self.memory_items = self.memory_items[-self.memory_capacity :]
        self._save_state()
        self._notify("info", f"Learned memory item: {item['id']}")
        return item

    def _delete_memory(self, mem_id: str) -> bool:
        before = len(self.memory_items)
        self.memory_items = [m for m in self.memory_items if str(m.get("id")) != str(mem_id)]
        changed = len(self.memory_items) < before
        if changed:
            self._save_state()
            self._notify("info", f"Deleted memory item: {mem_id}")
        return changed

    # ---------------- model-based responses ----------------
    def _build_outcome_explanation(self, rec: Dict[str, Any]) -> str:
        prompt = (
            "Explain this settled trade in plain language: why it likely won/lost, "
            "what indicator context mattered, and one caution for next trade."
        )
        context = {
            "allow_web_search": False,
            "record": _compact(rec),
            "latest_market_state": _compact(self.latest_market_state.get(rec.get("market") or "")),
            "recent_outcomes": _compact(list(self.recent_outcomes)[-40:]),
            "memory_items": _compact(self.memory_items[-80:]),
        }
        out = self._generate_with_continuation(prompt, context, max_passes=3)
        if isinstance(out, str) and out.strip():
            return out.strip()
        llm = get_last_llm_status()
        return f"Outcome recorded ({rec.get('outcome')}). LLM unavailable: {llm.get('error') or 'unknown'}."

    def _record_chat(self, role: str, text: str) -> None:
        try:
            if not text:
                return
            self.chat_history.append({"ts": _now_ts(), "role": role, "text": text})
        except Exception:
            pass

    def _runtime_status(self) -> Dict[str, Any]:
        now = _now_ts()
        tick_age = (now - int(self.last_tick_ts)) if self.last_tick_ts else None
        analysis_age = (now - int(self.last_analysis_event_ts)) if self.last_analysis_event_ts else None
        tick_state = "offline"
        if tick_age is not None:
            if tick_age <= 8:
                tick_state = "live"
            elif tick_age <= 30:
                tick_state = "stale"
        analysis_state = "idle"
        if analysis_age is not None:
            if analysis_age <= 10:
                analysis_state = "live"
            elif analysis_age <= 45:
                analysis_state = "stale"
        return {
            "tick_state": tick_state,
            "analysis_state": analysis_state,
            "tick_age_sec": tick_age,
            "analysis_age_sec": analysis_age,
        }

    def _looks_incomplete(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        if len(t) < 20:
            return False
        if t.endswith(("...", "…", ",", ":", ";", "-", "—")):
            return True
        if t[-1] not in ".!?)]}\"'":
            if len(t.split()) >= 8:
                return True
        tail_words = {"and", "or", "with", "of", "to", "for", "from", "that", "which", "because", "so", "but", "if", "when", "while", "then", "the", "a", "an", "is", "are"}
        words = t.lower().split()
        if words and words[-1] in tail_words:
            return True
        if t.count("**") % 2 == 1:
            return True
        return False

    def _generate_with_continuation(self, prompt: str, context: Dict[str, Any], max_passes: int = 4) -> Optional[str]:
        full = maybe_generate_with_openai(prompt, context)
        if not isinstance(full, str) or not full.strip():
            return None
        full = full.strip()
        continued = False
        for _ in range(max(0, max_passes - 1)):
            if not self._looks_incomplete(full):
                break
            cont_prompt = (
                "Continue the previous answer from exactly where it cut off. "
                "Do not restart. Complete all unfinished sentences.\n\n"
                f"Previous partial answer:\n{full}"
            )
            cont_ctx = dict(context)
            cont_ctx["allow_web_search"] = False
            cont = maybe_generate_with_openai(cont_prompt, cont_ctx)
            if not isinstance(cont, str) or not cont.strip():
                break
            c = cont.strip()
            if c in full:
                break
            full = (full + " " + c).strip()
            continued = True
        if self._looks_incomplete(full) and not continued:
            full = full + " [Response appears truncated. Ask: continue.]"
        return full

    def _last_assistant_message(self) -> Optional[str]:
        for row in reversed(self.chat_history):
            if str(row.get("role")) == "assistant":
                txt = str(row.get("text") or "").strip()
                if txt:
                    return txt
        return None

    # ---------------- public API ----------------
    def get_notifications(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._lock:
            n = max(1, min(int(limit or 30), 200))
            arr = list(self.notifications)
            return list(reversed(arr[-n:]))

    def get_learned_items(self, limit: int = 40) -> List[Dict[str, Any]]:
        with self._lock:
            n = max(1, min(int(limit or 40), 200))
            arr = list(self.learned_items)
            return list(reversed(arr[-n:]))

    def get_knowledge_items(self, limit: int = 60) -> List[Dict[str, Any]]:
        with self._lock:
            n = max(1, min(int(limit or 60), 300))
            arr = sorted(
                list(self.knowledge_items),
                key=lambda x: (int(x.get("importance") or 0), int(x.get("updated_ts") or 0)),
                reverse=True,
            )
            return arr[:n]

    def get_news_items(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            n = max(1, min(int(limit or 50), 200))
            arr = sorted(
                list(self.news_items),
                key=lambda x: (int(x.get("importance") or 0), int(x.get("updated_ts") or 0)),
                reverse=True,
            )
            return arr[:n]

    def clear_notifications(self) -> int:
        with self._lock:
            n = len(self.notifications)
            self.notifications.clear()
            try:
                with open(self.notifications_log, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                pass
            return n

    def stats_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            outcomes = list(self.recent_outcomes)
            total = len(outcomes)
            wins = sum(1 for r in outcomes if str(r.get("outcome")).upper() == "WIN")
            losses = sum(1 for r in outcomes if str(r.get("outcome")).upper() == "LOSS")
            wr = (wins / float(total)) if total > 0 else 0.0
            mem_used = len(self.memory_items)
            mem_pct = (mem_used / float(self.memory_capacity)) * 100.0 if self.memory_capacity > 0 else 0.0
            queued = len([t for t in self.tasks if t.get("status") == "queued"])
            running = len([t for t in self.tasks if t.get("status") == "running"])
            done = len([t for t in self.tasks if t.get("status") == "done"])
            failed = len([t for t in self.tasks if t.get("status") == "failed"])
            return {
                "total_settled_predictions": total,
                "wins": wins,
                "losses": losses,
                "win_rate": wr,
                "learned_items_count": len(self.learned_items),
                "knowledge_items_count": len(self.knowledge_items),
                "news_items_count": len(self.news_items),
                "memory_capacity": self.memory_capacity,
                "memory_used": mem_used,
                "memory_remaining": max(0, self.memory_capacity - mem_used),
                "memory_pct_used": mem_pct,
                "memory_pct_remaining": max(0.0, 100.0 - mem_pct),
                "tasks": {"queued": queued, "running": running, "done": done, "failed": failed},
                "latest_markets": _compact(self.latest_market_state),
                "market_tick_tail": _compact({k: list(v)[-15:] for k, v in self.market_tick_tail.items()}),
                "recent_outcomes": _compact(outcomes[-20:]),
                "llm": get_last_llm_status(),
                "runtime": self._runtime_status(),
                "research": {
                    "enabled": self.auto_research_enabled,
                    "interval_sec": self.research_interval_sec,
                    "last_ts": int(self.last_research_ts or 0),
                    "last_status": str(self.last_research_status or "idle"),
                    "last_error": str(self.last_research_error or ""),
                    "pause_until_ts": int(self.research_pause_until_ts or 0),
                    "cycles": int(self.total_research_cycles or 0),
                },
                "chat_history_size": len(self.chat_history),
                "total_tick_events": self.total_tick_events,
                "total_analysis_events": self.total_analysis_events,
            }

    def chat(self, user_message: str) -> str:
        msg = (user_message or "").strip()
        if not msg:
            return "Send a message."

        def done(text: str) -> str:
            with self._lock:
                self._record_chat("assistant", text)
                self._save_state()
            return text

        with self._lock:
            self._record_chat("user", msg)
            m = msg.lower()

            if m.startswith("learn about "):
                topic = msg[len("learn about ") :].strip()
                if not topic:
                    return done("Please provide a topic after 'learn about'.")
                t = self._add_task(f"Research and learn deeply about: {topic}", kind="learn", topic=topic)
                return done(f"Learning queued: {t['id']} on '{topic}'. I will notify you when it is complete.")

            if m.startswith("remember ") or m.startswith("learn "):
                txt = msg.split(" ", 1)[1].strip() if " " in msg else ""
                if not txt:
                    return done("Nothing to remember.")
                item = self._add_memory(txt, source="chat")
                st = self.stats_snapshot()
                return done(f"Saved as {item['id']}. Memory {st['memory_used']}/{st['memory_capacity']} ({st['memory_pct_used']:.1f}% used).")

            if m.startswith("forget ") or m.startswith("delete memory "):
                mem_id = msg.split(" ", 1)[1].replace("memory", "", 1).strip()
                if not mem_id:
                    return done("Provide memory id to delete.")
                ok = self._delete_memory(mem_id)
                return done(f"Deleted memory {mem_id}." if ok else f"Memory id not found: {mem_id}.")

            if m in ("list memories", "show memories", "memories"):
                if not self.memory_items:
                    return done("No memory items stored yet.")
                rows = [f"- {x.get('id')}: {x.get('text')}" for x in self.memory_items[-30:]]
                return done("Stored memories:\n" + "\n".join(rows))

            if m in ("clear memory", "clear memories", "delete memories"):
                self.memory_items = []
                self._save_state()
                return done("All memory items cleared.")

            if m.startswith("task:") or m.startswith("queue task:") or m.startswith("add task:"):
                text = msg.split(":", 1)[1].strip() if ":" in msg else ""
                if not text:
                    return done("Task text is empty.")
                low = text.lower()
                is_learning = ("learn" in low) or ("research" in low)
                t = self._add_task(text, kind=("learn" if is_learning else "task"), topic=(text if is_learning else None))
                return done(f"Task queued: {t['id']}. I will run tasks in order.")

            if ("did you learn" in m) or ("what did you learn" in m):
                done_tasks = [t for t in self.tasks if t.get("status") == "done"]
                running_tasks = [t for t in self.tasks if t.get("status") == "running"]
                if running_tasks:
                    rt = running_tasks[-1]
                    return done(f"I am still working on: {rt.get('text')} (task {rt.get('id')}).")
                if not self.learned_items and not done_tasks:
                    return done("No completed learning task yet. Use 'learn about ...' or 'task: ...'.")
                if self.learned_items:
                    li = self.learned_items[-1]
                    rows = [f"- {s.get('title')}: {s.get('takeaway') or _trim_text(s.get('learned'), max_len=120)}" for s in (li.get("subtopics") or [])[:5]]
                    return done(
                        f"Latest learning: {li.get('title')}.\n"
                        f"Summary: {li.get('summary')}\n"
                        f"Takeaway: {li.get('takeaway')}\n"
                        + ("\nSubtopics:\n" + "\n".join(rows) if rows else "")
                    )
                lt = done_tasks[-1]
                return done(f"Latest completed task: {lt.get('id')} - {lt.get('text')}")

            if m in ("learned", "show learned", "what have you learned", "learning library"):
                if not self.learned_items:
                    return done("No learned topics yet. Use 'learn about <topic>' to build the learning library.")
                rows = [f"- {x.get('id')}: {x.get('title')} ({len(x.get('subtopics') or [])} subtopics)" for x in self.learned_items[-20:]]
                return done("Learned topics:\n" + "\n".join(rows))

            if m in ("news", "latest news", "show news", "market news", "forex news", "deriv news"):
                items = self.get_news_items(limit=12)
                if not items:
                    return done("No news stored yet. Auto research will populate this shortly.")
                rows = []
                for n in items[:10]:
                    rows.append(
                        f"- [{n.get('importance')}] {n.get('title')} ({n.get('market') or 'market'})"
                    )
                return done("Top ranked news:\n" + "\n".join(rows))

            if m in ("knowledge", "show knowledge", "what do you know"):
                items = self.get_knowledge_items(limit=12)
                if not items:
                    return done("Knowledge library is still empty. Auto research is running and will fill it.")
                rows = [f"- [{k.get('importance')}] {k.get('title')}" for k in items[:10]]
                return done("Top knowledge in memory:\n" + "\n".join(rows))

            if m in ("strategies", "what strategies do you have", "show strategies", "list strategies"):
                items = self.get_knowledge_items(limit=220)
                strategies = [k for k in items if self._is_strategy_like(str(k.get("title") or ""), str(k.get("category") or ""), list(k.get("tags") or []))]
                if not strategies:
                    return done("No strategies indexed yet. Auto research is running and will add them.")
                rows = [f"- {k.get('title')} [{k.get('importance')}]" for k in strategies[:30]]
                return done("Saved strategies:\n" + "\n".join(rows))

            if m.startswith("teach strategy "):
                name = msg[len("teach strategy ") :].strip().lower()
                if not name:
                    return done("Provide a strategy name after 'teach strategy'.")
                items = self.get_knowledge_items(limit=260)
                cand = [k for k in items if name in str(k.get("title") or "").lower()]
                if not cand:
                    return done(f"I could not find strategy '{name}' in memory.")
                chosen = cand[0]
                teach_prompt = (
                    "Teach this strategy clearly in practical steps.\n"
                    "Include: setup conditions, entry, invalidation, and common mistakes to avoid."
                )
                teach_ctx = {
                    "allow_web_search": False,
                    "strategy": chosen,
                    "related_knowledge": _compact(cand[:5]),
                }
                out_t = self._generate_with_continuation(teach_prompt, teach_ctx, max_passes=4)
                if isinstance(out_t, str) and out_t.strip():
                    return done(out_t.strip())
                return done(f"Strategy: {chosen.get('title')}\nSummary: {chosen.get('summary')}\nTakeaway: {chosen.get('takeaway')}")

            if m in ("research status", "auto research status"):
                st = self.stats_snapshot()
                rs = st.get("research") or {}
                last_ts = int(rs.get("last_ts") or 0)
                pause_ts = int(rs.get("pause_until_ts") or 0)
                last_txt = datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat() if last_ts > 0 else "never"
                pause_txt = datetime.fromtimestamp(pause_ts, tz=timezone.utc).isoformat() if pause_ts > 0 else "none"
                return done(
                    f"Auto research: enabled={rs.get('enabled')}, interval={rs.get('interval_sec')}s, "
                    f"last_status={rs.get('last_status')}, cycles={rs.get('cycles')}, "
                    f"last_update={last_txt}, paused_until={pause_txt}."
                )

            if m in ("run research now", "refresh news", "refresh knowledge"):
                try:
                    self._auto_research_once()
                    return done("Auto research run completed.")
                except Exception as e:
                    return done(f"Auto research run failed: {str(e)}")

            if m in ("resume research", "resume auto research", "unpause research"):
                with self._lock:
                    self.research_pause_until_ts = 0
                    self.last_research_status = "idle"
                    self.last_research_error = ""
                    self._save_state()
                return done("Auto research resumed. I will run on the next cycle.")

            if m in ("tasks", "show tasks", "task status"):
                if not self.tasks:
                    return done("No tasks yet.")
                rows = [f"- {t.get('id')} [{t.get('status')}] {t.get('text')}" for t in self.tasks[-20:]]
                return done("Task queue:\n" + "\n".join(rows))

            if m in ("clear tasks", "delete tasks"):
                self.tasks = []
                self._save_state()
                return done("All tasks cleared.")

            if m in ("continue", "go on", "continue please", "finish that", "finish", "keep going"):
                prev = self._last_assistant_message()
                if not prev:
                    return done("There is no previous assistant response to continue.")
                snapshot = self.stats_snapshot()
                cont_context = {
                    "allow_web_search": False,
                    "stats": snapshot,
                    "runtime": snapshot.get("runtime"),
                    "memory_items": _compact(self.memory_items[-120:]),
                    "learned_items": _compact(self.learned_items[-60:]),
                    "knowledge_items": _compact(self.get_knowledge_items(limit=80)),
                    "news_items": _compact(self.get_news_items(limit=40)),
                    "recent_chat_history": _compact(list(self.chat_history)[-25:]),
                }
                cont_prompt = (
                    "Continue your previous answer naturally from exactly where it stopped. "
                    "Do not restart from the beginning.\n\n"
                    f"Previous answer:\n{prev}"
                )
                outc = self._generate_with_continuation(cont_prompt, cont_context, max_passes=5)
                if isinstance(outc, str) and outc.strip():
                    cleaned = outc.strip()
                    if cleaned.startswith(prev):
                        cleaned = cleaned[len(prev) :].strip()
                    return done(cleaned or "Done.")
                llm = get_last_llm_status()
                return done(f"I could not continue right now. LLM error: {llm.get('error')}.")

            if ("where currently stand" in m) or ("current stand" in m) or ("running or not" in m) or (m in ("status", "current status")):
                st = self.stats_snapshot()
                r = st.get("runtime") or {}
                return done(
                    f"Current status: ticks={r.get('tick_state')} (age={r.get('tick_age_sec')}s), "
                    f"analysis={r.get('analysis_state')} (age={r.get('analysis_age_sec')}s), "
                    f"win_rate={(st.get('win_rate',0.0)*100):.1f}% over {st.get('total_settled_predictions',0)} settled trades, "
                    f"tasks queued/running={st.get('tasks',{}).get('queued',0)}/{st.get('tasks',{}).get('running',0)}."
                )

            snapshot = self.stats_snapshot()
            context = {
                "allow_web_search": True,
                "user_message": msg,
                "stats": snapshot,
                "runtime": snapshot.get("runtime"),
                "memory_items": _compact(self.memory_items[-200:]),
                "learned_items": _compact(self.learned_items[-120:]),
                "knowledge_items": _compact(self.get_knowledge_items(limit=140)),
                "news_items": _compact(self.get_news_items(limit=80)),
                "tasks": _compact(self.tasks[-80:]),
                "latest_markets": _compact(self.latest_market_state),
                "recent_ticks": _compact({k: list(v)[-20:] for k, v in self.market_tick_tail.items()}),
                "recent_outcomes": _compact(list(self.recent_outcomes)[-120:]),
                "recent_notifications": _compact(self.get_notifications(limit=30)),
                "recent_chat_history": _compact(list(self.chat_history)[-20:]),
            }

        prompt = (
            f"User message: {msg}\n"
            "Respond conversationally like a strong assistant.\n"
            "Always complete sentences fully (no cut-off endings).\n"
            "If user asks trading questions: interpret trend + indicators in plain language, then give practical next action.\n"
            "If user asks learning/research questions: provide structured explanation and concise teaching points.\n"
            "Use learned_items context when available, and teach from it directly.\n"
            "Use news_items and knowledge_items when relevant, and prioritize high-importance entries.\n"
            "If uncertainty exists, state it explicitly.\n"
        )
        out = self._generate_with_continuation(prompt, context, max_passes=5)
        if isinstance(out, str) and out.strip():
            return done(out.strip())
        llm = get_last_llm_status()
        return done(
            "I could not get a model response right now. "
            f"Provider={llm.get('provider')} model={llm.get('model')} error={llm.get('error')}."
        )


assistant_engine = HeroAssistantEngine()
assistant_blueprint = Blueprint("hero_assistant", __name__)


@assistant_blueprint.route("/assistant/health", methods=["GET"])
def assistant_health():
    return jsonify(
        {
            "ok": True,
            "service": "hero_assistant",
            "ts": _now_ts(),
            "memory_dir": assistant_engine.base_dir,
            "stats": assistant_engine.stats_snapshot(),
        }
    )


@assistant_blueprint.route("/assistant/notifications", methods=["GET"])
def assistant_notifications():
    try:
        limit = int(request.args.get("limit") or 30)
    except Exception:
        limit = 30
    return jsonify({"ok": True, "notifications": assistant_engine.get_notifications(limit=limit)})


@assistant_blueprint.route("/assistant/notifications/clear", methods=["POST"])
def assistant_notifications_clear():
    cleared = assistant_engine.clear_notifications()
    return jsonify({"ok": True, "cleared": cleared})


@assistant_blueprint.route("/assistant/stats", methods=["GET"])
def assistant_stats():
    return jsonify({"ok": True, "stats": assistant_engine.stats_snapshot()})


@assistant_blueprint.route("/assistant/tasks", methods=["GET"])
def assistant_tasks():
    return jsonify({"ok": True, "tasks": assistant_engine.tasks[-200:]})


@assistant_blueprint.route("/assistant/memories", methods=["GET"])
def assistant_memories():
    return jsonify({"ok": True, "memories": assistant_engine.memory_items[-500:]})


@assistant_blueprint.route("/assistant/learned", methods=["GET"])
def assistant_learned():
    try:
        limit = int(request.args.get("limit") or 40)
    except Exception:
        limit = 40
    return jsonify({"ok": True, "learned": assistant_engine.get_learned_items(limit=limit)})


@assistant_blueprint.route("/assistant/knowledge", methods=["GET"])
def assistant_knowledge():
    try:
        limit = int(request.args.get("limit") or 60)
    except Exception:
        limit = 60
    return jsonify({"ok": True, "knowledge": assistant_engine.get_knowledge_items(limit=limit)})


@assistant_blueprint.route("/assistant/news", methods=["GET"])
def assistant_news():
    try:
        limit = int(request.args.get("limit") or 50)
    except Exception:
        limit = 50
    return jsonify({"ok": True, "news": assistant_engine.get_news_items(limit=limit)})


@assistant_blueprint.route("/assistant/chat", methods=["POST"])
def assistant_chat():
    obj = request.get_json(force=True, silent=True) or {}
    if not isinstance(obj, dict):
        return jsonify({"ok": False, "error": "expected object"}), 400
    msg = str(obj.get("message") or "").strip()
    if not msg:
        return jsonify({"ok": False, "error": "missing message"}), 400
    reply = assistant_engine.chat(msg)
    return jsonify(
        {
            "ok": True,
            "reply": reply,
            "notifications": assistant_engine.get_notifications(limit=10),
            "stats": assistant_engine.stats_snapshot(),
        }
    )
