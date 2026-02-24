#!/usr/bin/env python3
"""
openai_api_placeholder.py

LLM adapter for Hero Assistant.
Primary provider is Gemini. OpenAI path is kept for compatibility.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Persistent key ring fallback (used when env/file keys are not provided).
DEFAULT_GEMINI_KEYS = [
    "AIzaSyAZl7E_UK-rzFi4Bkv4VfrnnmQiF_wjiuw",
    "AIzaSyAxwKQZWvPc1xuMcMMEfFqgUBpN1qxzJw4",
]
DEFAULT_GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]

_LAST_LLM_STATUS: Dict[str, Any] = {"provider": None, "model": None, "ok": False, "error": "not_called", "ts": 0}


def get_last_llm_status() -> Dict[str, Any]:
    return dict(_LAST_LLM_STATUS)


def _set_status(provider: str, model: Optional[str], ok: bool, error: Optional[str]) -> None:
    _LAST_LLM_STATUS.update(
        {
            "provider": provider,
            "model": model,
            "ok": bool(ok),
            "error": error if error else None,
            "ts": int(time.time()),
        }
    )


def _truthy(v: Any) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _split_and_clean_csv(v: str) -> List[str]:
    out: List[str] = []
    for x in str(v or "").split(","):
        s = str(x or "").strip()
        if s:
            out.append(s)
    return out


def _gemini_keys_file_path() -> str:
    return os.path.expanduser(os.environ.get("HERO_ASSISTANT_GEMINI_KEYS_FILE", "~/.hero_logs/gemini_keys.json"))


def _load_gemini_keys_from_file(path: str) -> List[str]:
    try:
        if not path or not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            keys = obj.get("keys")
            if isinstance(keys, list):
                return [str(k).strip() for k in keys if str(k).strip()]
            single = obj.get("gemini_api_key")
            if isinstance(single, str) and single.strip():
                return [single.strip()]
        if isinstance(obj, list):
            return [str(k).strip() for k in obj if str(k).strip()]
    except Exception:
        return []
    return []


def _gemini_api_keys() -> List[str]:
    out: List[str] = []
    seen = set()

    def _add_many(rows: List[str]) -> None:
        for k in rows:
            kk = str(k or "").strip()
            if not kk:
                continue
            lk = kk.lower()
            if lk in seen:
                continue
            seen.add(lk)
            out.append(kk)

    # Highest priority: explicit multi-key env
    _add_many(_split_and_clean_csv(os.environ.get("HERO_ASSISTANT_GEMINI_KEYS") or ""))
    # Next: standard single-key env
    env_single = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if env_single:
        _add_many([env_single])
    # Next: persisted local keys file
    _add_many(_load_gemini_keys_from_file(_gemini_keys_file_path()))
    # Fallback: in-code key ring
    _add_many(DEFAULT_GEMINI_KEYS)
    return out


def _compact_context(context: Dict[str, Any], max_len: int = 16000) -> str:
    safe_context = context if isinstance(context, dict) else {}
    compact = json.dumps(safe_context, ensure_ascii=True, separators=(",", ":"), default=str)
    if len(compact) > max_len:
        compact = compact[:max_len] + "...[truncated]"
    return compact


def _extract_openai_text(raw_json: str) -> Optional[str]:
    try:
        obj = json.loads(raw_json)
        choices = obj.get("choices") if isinstance(obj, dict) else None
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str) and content.strip():
                return content.strip()
    except Exception:
        return None
    return None


def _extract_gemini_text(raw_json: str) -> Optional[str]:
    try:
        obj = json.loads(raw_json)
        cands = obj.get("candidates") if isinstance(obj, dict) else None
        if isinstance(cands, list) and cands:
            first = cands[0] if isinstance(cands[0], dict) else {}
            content = first.get("content") if isinstance(first, dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if isinstance(parts, list):
                texts: List[str] = []
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        t = p.get("text").strip()
                        if t:
                            texts.append(t)
                if texts:
                    return "\n".join(texts).strip()
    except Exception:
        return None
    return None


def _post_json(url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: int = 45) -> Tuple[Optional[str], Optional[str]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace"), None
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(e)
        return None, f"HTTP {getattr(e, 'code', 'error')}: {detail[:280]}"
    except urllib.error.URLError as e:
        return None, f"URL error: {str(e)}"
    except TimeoutError:
        return None, "timeout"
    except Exception as e:
        return None, str(e)


def _get_json(url: str, headers: Dict[str, str], timeout: int = 20) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj, None
            return None, "invalid_json_shape"
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(e)
        return None, f"HTTP {getattr(e, 'code', 'error')}: {detail[:220]}"
    except urllib.error.URLError as e:
        return None, f"URL error: {str(e)}"
    except TimeoutError:
        return None, "timeout"
    except Exception as e:
        return None, str(e)


def _normalize_gemini_root(base_url: str) -> str:
    b = (base_url or "").strip().rstrip("/")
    if not b:
        b = "https://generativelanguage.googleapis.com"
    for suffix in ("/v1beta", "/v1"):
        if b.endswith(suffix):
            b = b[: -len(suffix)]
    return b


def _discover_gemini_models(api_key: str, base_root: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for ver in ("v1beta", "v1"):
        url = f"{base_root}/{ver}/models?key={urllib.parse.quote(api_key)}"
        obj, _err = _get_json(url, headers={"Accept": "application/json"})
        if _is_auth_or_permission_error(_err or ""):
            return []
        if not isinstance(obj, dict):
            continue
        arr = obj.get("models")
        if not isinstance(arr, list):
            continue
        for item in arr:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            methods = item.get("supportedGenerationMethods")
            supports = isinstance(methods, list) and any(str(x).strip().lower() == "generatecontent" for x in methods)
            if not supports:
                continue
            k = name.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(name)
    return out


def _gemini_api_versions() -> List[str]:
    env_ver = (os.environ.get("GEMINI_API_VERSION") or "").strip().lower()
    if env_ver in ("v1beta", "v1"):
        return [env_ver]
    return ["v1beta", "v1"]


def _as_model_path(model_name: str) -> str:
    m = str(model_name or "").strip()
    if m.startswith("models/"):
        return m
    return f"models/{m}"


def _model_short_name(model_name: str) -> str:
    m = _as_model_path(model_name)
    return m.split("/", 1)[1] if "/" in m else m


def _should_try_next_model(err: str) -> bool:
    e = (err or "").lower()
    return ("not found" in e) or ("404" in e) or ("not supported" in e)


def _is_auth_or_permission_error(err: str) -> bool:
    e = (err or "").lower()
    return (
        ("http 401" in e)
        or ("http 403" in e)
        or ("permission_denied" in e)
        or ("unauthenticated" in e)
        or ("invalid api key" in e)
        or ("api key was reported as leaked" in e)
        or (("api key" in e) and ("forbidden" in e))
    )


def _is_quota_or_rate_error(err: str) -> bool:
    e = (err or "").lower()
    return (
        ("http 429" in e)
        or ("resource_exhausted" in e)
        or ("rate limit" in e)
        or ("quota" in e)
        or ("too many requests" in e)
    )


def _call_openai(user_message: str, context: Dict[str, Any]) -> Optional[str]:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        _set_status("openai", None, False, "missing_api_key")
        return None

    model = (os.environ.get("HERO_ASSISTANT_MODEL") or "gpt-4.1-mini").strip()
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    compact_context = _compact_context(context)
    system_prompt = (
        "You are HeroX trading assistant. Be conversational, practical, and educational. "
        "Use the provided context; do not claim certainty."
    )
    user_prompt = (
        f"User question:\n{user_message}\n\n"
        f"Context JSON:\n{compact_context}\n\n"
        "Respond in plain language with: what you observe, why it matters, and what to avoid."
    )
    body = {
        "model": model,
        "temperature": 0.25,
        "max_tokens": 1400,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    raw, err = _post_json(
        endpoint,
        body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    if not raw:
        _set_status("openai", model, False, err or "no_response")
        return None
    txt = _extract_openai_text(raw)
    if txt:
        _set_status("openai", model, True, None)
        return txt
    _set_status("openai", model, False, "empty_text")
    return None


def _gemini_models(api_key: str, base_root: str) -> List[str]:
    env_val = (os.environ.get("HERO_ASSISTANT_GEMINI_MODEL") or "").strip()
    out: List[str] = []
    seen = set()

    if env_val:
        for m in [x.strip() for x in env_val.split(",") if x.strip()]:
            k = m.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(m)

    for m in _discover_gemini_models(api_key, base_root):
        k = m.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(m)

    for m in DEFAULT_GEMINI_MODELS:
        k = m.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(m)
    return out


def _call_gemini(user_message: str, context: Dict[str, Any]) -> Optional[str]:
    api_keys = _gemini_api_keys()
    if not api_keys:
        _set_status("gemini", None, False, "missing_api_key")
        return None

    base_root = _normalize_gemini_root(os.environ.get("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com")
    compact_context = _compact_context(context)
    allow_web = bool(context.get("allow_web_search", True))
    prompt = (
        "You are HeroX trading assistant.\n"
        "Be conversational and educational.\n"
        "Explain clearly: current trend, why last trade won/lost, risk signals, and next caution.\n"
        "Do not just dump raw numbers; interpret them.\n\n"
        f"User question:\n{user_message}\n\n"
        f"Context JSON:\n{compact_context}"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.25, "maxOutputTokens": 2048},
    }
    if allow_web:
        body["tools"] = [{"google_search": {}}]

    attempts: List[str] = []
    last_err: Optional[str] = None
    versions = _gemini_api_versions()

    for key_idx, api_key in enumerate(api_keys):
        models = _gemini_models(api_key, base_root)
        if not models:
            last_err = "no_models"
            attempts.append(f"k{key_idx+1}:no_models")
            continue

        switch_key = False
        for model in models:
            short = _model_short_name(model)
            for ver in versions:
                endpoint = f"{base_root}/{ver}/{_as_model_path(model)}:generateContent?key={urllib.parse.quote(api_key)}"
                raw, err = _post_json(endpoint, body, headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
                if not raw:
                    # Retry same model/version without web tool if this API/version rejects tools payload.
                    low = (err or "").lower()
                    if allow_web and (("tools" in low) or ("google_search" in low) or ("unknown field" in low) or ("invalid argument" in low)):
                        body_no_tools = dict(body)
                        body_no_tools.pop("tools", None)
                        raw2, err2 = _post_json(endpoint, body_no_tools, headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
                        if raw2:
                            txt2 = _extract_gemini_text(raw2)
                            if txt2:
                                _set_status("gemini", f"k{key_idx+1}:{short}@{ver}", True, None)
                                return txt2
                        err = err2 or err
                    last_err = err or "no_response"
                    attempts.append(f"k{key_idx+1}:{short}@{ver}:{(last_err or 'error')[:70]}")

                    # Fast switch to next key for auth/quota failures.
                    if _is_auth_or_permission_error(last_err) or _is_quota_or_rate_error(last_err):
                        switch_key = True
                        break
                    if _should_try_next_model(last_err):
                        break
                    continue

                txt = _extract_gemini_text(raw)
                if txt:
                    _set_status("gemini", f"k{key_idx+1}:{short}@{ver}", True, None)
                    return txt
                last_err = "empty_text"
                attempts.append(f"k{key_idx+1}:{short}@{ver}:empty_text")
            if switch_key:
                break

    err_txt = (last_err or "failed")
    if attempts:
        err_txt = f"{err_txt} | attempts: " + ", ".join(attempts[-6:])
    _set_status("gemini", f"keys={len(api_keys)}", False, err_txt)
    return None


def maybe_generate_with_openai(user_message: str, context: Dict[str, Any]) -> Optional[str]:
    """
    Compatibility function name used by hero_assistant.py.
    Uses Gemini by default.
    """
    enabled_env = os.environ.get("HERO_ASSISTANT_LLM_ENABLED")
    if enabled_env is not None and not _truthy(enabled_env):
        _set_status("none", None, False, "disabled")
        return None

    provider = (os.environ.get("HERO_ASSISTANT_LLM_PROVIDER") or "gemini").strip().lower()
    if provider == "openai":
        openai_enabled = os.environ.get("HERO_ASSISTANT_OPENAI_ENABLED")
        if openai_enabled is not None and not _truthy(openai_enabled):
            _set_status("openai", None, False, "openai_disabled")
            return None
        return _call_openai(user_message, context)

    # default
    return _call_gemini(user_message, context)
