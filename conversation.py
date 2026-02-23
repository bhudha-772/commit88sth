# conversation.py
from __future__ import annotations
import re
import time
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memory_local import MemoryStore  # only for type checking

class ConversationManager:
    def __init__(self, memory: Optional["MemoryStore"] = None):
        # user_id -> metadata
        self.users: Dict[int, Dict] = {}
        self.memory = memory

        self._utterances = {
            "greet": ["Hi there!", "Hey! How's it going?", "Hello — nice to meet you."],
            "ask_how": ["I'm a program, so 'good' is my baseline — how are you?", "All systems nominal! How about you?"],
            "about_bot": ["I'm HeroX — a lightweight assistant. I can chat, remember short notes, and launch a local dashboard."],
            "ask_future": ["I want to be useful, reliable, and help people build tools. Where would you like to be in a year?"],
            "thanks": ["You're welcome!", "Anytime!", "Glad to help."],
            "bye": ["Goodbye — talk soon!", "See you later!"],
            "fallback": [
                "Sorry — I can't do that. I can chat and I support commands like /opendashboard /closedashboard /dashboardurl /ingesturl /recall /memorystats.",
                "I don't have a tool for that. I can hold conversations or manage the dashboard; try asking me something else."
            ],
        }

    def _pick(self, key: str) -> str:
        lst = self._utterances.get(key, [])
        if not lst:
            return ""
        idx = int(time.time()) % len(lst)
        return lst[idx]

    def _ensure_user(self, user_id: int):
        if user_id not in self.users:
            self.users[user_id] = {"name": None, "last_intent": None, "last_seen": time.time(), "mood": None}

    def _detect_intent(self, text: str):
        t = (text or "").strip().lower()
        if re.search(r"\b(hi|hello|hey|yo|sup)\b", t):
            return "greet", None
        if re.search(r"\b(who\s+are\s+you|what\s+are\s+you|tell\s+me\s+about\s+yourself)\b", t):
            return "about_bot", None
        if re.search(r"\b(how\s+are\s+you|how\s+are\s+things)\b", t):
            return "ask_how", None
        m = re.search(r"\bmy\s+name\s+is\s+([A-Za-z0-9 _-]+)\b", text or "", flags=re.I)
        if m:
            return "tell_name", m.group(1).strip()
        if re.search(r"\b(what\s+is\s+my\s+name|do\s+you\s+remember\s+me)\b", t):
            return "query_name", None
        if re.search(r"\b(what\s+do\s+you\s+want|where\s+would\s+you\s+want)\b", t):
            return "ask_future", None
        if re.search(r"\b(thank(s| you)|thx)\b", t):
            return "thanks", None
        if re.search(r"\b(bye|goodbye|see\s+you)\b", t):
            return "bye", None
        # "tell me about X" or "what do you know about X"
        m = re.match(r"^(tell me about|what do you know about)\s+(.+)$", t)
        if m:
            return "query_memory", m.group(2).strip()
        if len(t.split()) <= 3:
            if re.search(r"\b(good|fine|okay|ok|great|bad|sad|happy)\b", t):
                return "mood_update", t
        # fallback
        return "unknown", None

    def handle_message(self, user_id: int, text: str) -> str:
        self._ensure_user(user_id)
        intent, payload = self._detect_intent(text or "")
        self.users[user_id]["last_seen"] = time.time()
        self.users[user_id]["last_intent"] = intent

        if intent == "greet":
            return f"{self._pick('greet')} You can tell me your name (\"my name is ...\") or ask me to 'tell me about <topic>'."
        if intent == "about_bot":
            return " ".join([self._pick("about_bot"), "I also keep a local memory you can query with 'tell me about <topic>' or use /recall <query>."])
        if intent == "ask_how":
            return self._pick("ask_how")
        if intent == "tell_name" and payload:
            name = payload.strip().split()[0][:40]
            self.users[user_id]["name"] = name
            return f"Nice to meet you, {name}! I'll remember that in this session."
        if intent == "query_name":
            name = self.users[user_id].get("name")
            if name:
                return f"I remember your name: {name}."
            return "I don't know your name yet — tell me by saying 'my name is ...'."
        if intent == "ask_future":
            return self._pick("ask_future")
        if intent == "mood_update":
            mood_word = (payload or text).strip().split()[0]
            self.users[user_id]["mood"] = mood_word
            return f"I'm glad to hear you're {mood_word}. Want to tell me more?"
        if intent == "thanks":
            return self._pick("thanks")
        if intent == "bye":
            return self._pick("bye")
        if intent == "query_memory":
            topic = payload or ""
            if self.memory:
                rows = self.memory.search(topic, limit=6)
                if rows:
                    pieces = []
                    for rid, src, ts, snippet in rows:
                        ts_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
                        pieces.append(f"- ({src}) {snippet[:160]}  [{ts_str}]")
                    return f"I found {len(rows)} item(s) about \"{topic}\":\n" + "\n".join(pieces)
                else:
                    total_db = self.memory.count()
                    return f"I couldn't find anything about \"{topic}\" in memory. Currently memory holds {total_db} items."
            else:
                return "Memory is not available in this instance."
        if intent == "unknown":
            # fallback: ask follow-up question or refer to commands
            return self._pick("fallback")
        # default fallback
        return self._pick("fallback")
