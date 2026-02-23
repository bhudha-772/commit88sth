#!/usr/bin/env python3
"""
higher_lower_trade.py (fixed, cleaned, server-token aware)

Provides:
 - Flask endpoints: /simulate, /trade, /signal, /status
 - A background asyncio DerivAPI client (AsyncDerivClient) that runs on a thread
 - Robust token resolution (env vars preferred; fallback to asking hero_service via /control/get_server_token)
 - Detailed logging and structured TRADE_LOG file entries

Usage:
 - Import as a module and register hl_bp blueprint in hero_service.py:
     import higher_lower_trade
     app.register_blueprint(higher_lower_trade.hl_bp)
 - Or run standalone: python higher_lower_trade.py  (standalone will use port 5001 to avoid conflict)
"""
import os
import time
import json
import threading
import asyncio
import traceback
import concurrent.futures
import urllib.request as _ureq
import urllib.error as _uerr
import ssl as _ssl
from datetime import datetime
from typing import Any, Dict, Optional
from decimal import Decimal, ROUND_HALF_UP

# Flask
from flask import Flask, request, jsonify, Blueprint as _Blueprint

# logging
import logging
LOG_FILE = os.environ.get("TRADE_LOG", "trades.log")
LOG_DEBUG_FILE = os.environ.get("TRADE_DEBUG_LOG", "trades_debug.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("higher_lower_trade")
try:
    fh = logging.FileHandler(LOG_DEBUG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(fh)
except Exception:
    pass

# try to import python-deriv-api
try:
    from deriv_api import DerivAPI
except Exception as exc:
    logger.exception("python_deriv_api not installed or import failed: %s", exc)
    raise RuntimeError("python_deriv_api must be installed (pip install python_deriv_api)") from exc

# ---------------- Config ----------------
DEFAULT_APP_ID = int(os.environ.get("DERIV_APP_ID", "71710"))
# IMPORTANT: No hard-coded token here. Allow environment or server-provided tokens.
DEFAULT_TOKEN_DEMO = os.environ.get("DERIV_TOKEN") or None
DEFAULT_TOKEN_REAL = os.environ.get("DERIV_TOKEN_REAL") or None
TRADE_LOG = os.environ.get("TRADE_LOG", "trades.log")

# stake / currency defaults
DEFAULT_STAKE = float(os.environ.get("DEFAULT_STAKE", "1.0"))
MIN_STAKE = float(os.environ.get("MIN_STAKE", "0.35"))
DEFAULT_CURRENCY = os.environ.get("DERIV_CURRENCY", "USD")

# timeouts and polling
PROPOSAL_TIMEOUT = float(os.environ.get("PROPOSAL_TIMEOUT", "25"))
BUY_TIMEOUT = float(os.environ.get("BUY_TIMEOUT", "10"))
SETTLEMENT_POLL_INTERVAL = float(os.environ.get("SETTLEMENT_POLL_INTERVAL", "0.6"))
SETTLEMENT_POLL_TIMEOUT = float(os.environ.get("SETTLEMENT_POLL_TIMEOUT", "90"))

# Optional: base URL to hero_service (used to fetch server-side token). If hero_service provides
# /control/get_server_token?mode=demo|real, we can request tokens from it.
HERO_SERVICE_BASE = os.environ.get("HERO_SERVICE_BASE", "http://127.0.0.1:5000")

# ---------------- App (for standalone run) ----------------
app = Flask(__name__)

# ---------------- Helpers ----------------
def _normalize_direction(direction: str) -> str:
    d = (direction or "").strip().lower()
    if d in ("higher", "call", "buy", "up"):
        return "CALL"
    if d in ("lower", "put", "sell", "down"):
        return "PUT"
    raise ValueError("invalid direction; use 'higher' or 'lower'")

def _build_proposal_payload(symbol: str, direction: str, stake: float, barrier: str, duration: int, currency: Optional[str] = None) -> Dict[str, Any]:
    """
    Build a safe proposal payload. Use Decimal for stake rounding and return 'amount' as a numeric float.
    Returning a numeric value ensures the deriv websocket receives an actual number (not a string),
    preventing the API from interpreting it as 0 in some cases.
    """
    try:
        # Convert stake to Decimal via str to preserve what's passed in
        amt = Decimal(str(stake))
    except Exception:
        amt = Decimal(str(DEFAULT_STAKE))

    # enforce minimum
    if amt < Decimal(str(MIN_STAKE)):
        amt = Decimal(str(MIN_STAKE))

    # quantize to two decimal places (round half up)
    amt = amt.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    # convert to float for the payload (DERIV websocket prefers numeric)
    amt_float = float(amt)

    payload = {
        "proposal": 1,
        "symbol": symbol,
        "amount": amt_float,   # numeric 0.35 instead of string "0.35"
        "basis": "stake",
        "contract_type": direction,
        "duration": int(duration),
        "duration_unit": "t",
        "barrier": barrier,
        "currency": currency,
        "product_type": "basic",
    }
    return {k: v for k, v in payload.items() if v is not None}

def _extract_currency_from_auth(auth_resp: dict) -> Optional[str]:
    try:
        return auth_resp.get("authorize", {}).get("currency") or auth_resp.get("authorize", {}).get("currency_code")
    except Exception:
        return None

def _extract_proposal_id(proposal_resp: dict) -> Optional[str]:
    try:
        p = proposal_resp.get("proposal") or proposal_resp
        if isinstance(p, dict):
            return p.get("id") or p.get("proposal_id")
        if isinstance(proposal_resp, dict):
            return proposal_resp.get("id")
    except Exception:
        pass
    return None

def _extract_ask_price(proposal_resp: dict) -> Optional[float]:
    try:
        p = proposal_resp.get("proposal") or proposal_resp
        if isinstance(p, dict):
            if "ask_price" in p and p["ask_price"] not in (None, ""):
                return float(p["ask_price"])
            if "display_value" in p and p["display_value"] not in (None, ""):
                try:
                    return float(p["display_value"])
                except Exception:
                    pass
            if "payout" in p and p["payout"] not in (None, ""):
                try:
                    return float(p["payout"])
                except Exception:
                    pass
    except Exception:
        pass
    return None

def _log_trade(entry: dict):
    """
    Append structured JSON to TRADE_LOG file and log to standard logger.
    """
    try:
        e = dict(entry)
        e["ts"] = datetime.utcnow().isoformat() + "Z"
        with open(TRADE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(e, default=str) + "\n")
    except Exception:
        logger.exception("failed to write to TRADE_LOG")
    try:
        t = entry.get("type", "log")
        if entry.get("error") or entry.get("result") == "loss":
            logger.error("TRADE_LOG %s %s", t, entry)
        else:
            logger.info("TRADE_LOG %s %s", t, entry)
    except Exception:
        logger.debug("TRADE_LOG write fallback", exc_info=True)

# Fetch server-side token helper (mirrors approach in differ_trade_check)
def fetch_server_token(mode: str, hero_base: str = None, timeout: float = 2.0) -> Optional[str]:
    """
    Try to GET full token from hero_service endpoint /control/get_server_token?mode=<mode>.
    Returns token string or None.
    """
    try:
        if not mode or mode not in ("demo", "real"):
            return None
        if not hero_base:
            hero_base = HERO_SERVICE_BASE
        hero_base = hero_base.rstrip("/")
        url = f"{hero_base}/control/get_server_token?mode={mode}"
        req = _ureq.Request(url, method="GET")
        # if using http, context is ignored; safe to use default context
        ctx = _ssl.create_default_context()
        with _ureq.urlopen(req, timeout=timeout, context=ctx) as resp:
            if resp.status != 200:
                return None
            raw = resp.read().decode("utf-8")
            try:
                jo = json.loads(raw or "{}")
            except Exception:
                jo = {}
            # /control/get_server_token typically returns {"ok":True, "token": "..."}
            if jo.get("ok") and jo.get("token"):
                return jo.get("token")
            # older shapes
            if jo.get("ok") and jo.get("present") and jo.get("token"):
                return jo.get("token")
    except Exception:
        logger.debug("fetch_server_token failed (ignored)", exc_info=True)
        return None
    return None

def resolve_token(mode: str) -> (Optional[str], str):
    """
    Returns (token or None, source_string)
    Prefer environment tokens, then ask hero_service endpoint.
    """
    mode = (mode or "demo").lower()
    if mode not in ("demo", "real"):
        mode = "demo"
    if mode == "real" and DEFAULT_TOKEN_REAL:
        return DEFAULT_TOKEN_REAL, "env_real"
    if mode == "demo" and DEFAULT_TOKEN_DEMO:
        return DEFAULT_TOKEN_DEMO, "env_demo"
    # ask server
    try:
        tk = fetch_server_token(mode)
        if tk:
            return tk, "server"
    except Exception:
        logger.debug("resolve_token: fetch_server_token failed", exc_info=True)
    return None, "none"

# ---------------- Background Deriv client ----------------
class AsyncDerivClient:
    """
    Background asyncio loop with a DerivAPI instance.
    Provides .run(coro, timeout) and .submit(coro_fn, *args) helpers.
    """
    def __init__(self, app_id: int = DEFAULT_APP_ID, demo_token: Optional[str] = DEFAULT_TOKEN_DEMO, real_token: Optional[str] = DEFAULT_TOKEN_REAL):
        self.app_id = app_id
        self.demo_token = demo_token
        self.real_token = real_token
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.api = None
        self.currency: Optional[str] = None
        self.last_authorize_response: Optional[dict] = None
        self.last_authorize_ts: Optional[float] = None
        self._ready = threading.Event()
        self._start_background_loop()

    def _start_background_loop(self):
        logger.debug("starting background loop thread")
        def _run():
            try:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                logger.debug("background asyncio loop created")
                self.loop.run_forever()
            except Exception as e:
                logger.exception("AsyncDerivClient loop error: %s", e)
        self.thread = threading.Thread(target=_run, name="async-deriv-loop", daemon=True)
        self.thread.start()

        timeout = 5.0
        waited = 0.0
        while self.loop is None and waited < timeout:
            time.sleep(0.05)
            waited += 0.05

        if self.loop is None:
            logger.error("AsyncDerivClient failed to start loop within timeout")
            self._ready.set()
            return

        fut = asyncio.run_coroutine_threadsafe(self._create_api(), self.loop)
        try:
            fut.result(timeout=15.0)
            logger.info("AsyncDerivClient created api (or attempted to)")
        except Exception as e:
            logger.exception("AsyncDerivClient create_api failed: %s", e)
        finally:
            self._ready.set()

    async def _create_api(self):
        try:
            if getattr(self, "api", None) is not None:
                try:
                    close_m = getattr(self.api, "close", None)
                    if close_m:
                        maybe = close_m()
                        if asyncio.iscoroutine(maybe):
                            await maybe
                except Exception:
                    # some DerivAPI versions don't expose close; tolerate failures
                    logger.exception("previous api close failed")
            logger.info("Creating DerivAPI instance (app_id=%s)", self.app_id)
            self.api = DerivAPI(app_id=self.app_id)
            # Try initial authorize only if a token is provided in env
            token = self.demo_token
            if token:
                try:
                    logger.debug("attempting initial authorize with demo token (background)")
                    auth = await asyncio.wait_for(self.api.authorize(token), timeout=12.0)
                    self.last_authorize_response = auth
                    self.last_authorize_ts = time.time()
                    cc = _extract_currency_from_auth(auth) if isinstance(auth, dict) else None
                    if cc:
                        self.currency = cc
                        logger.info("initial authorize succeeded, currency=%s", cc)
                    else:
                        logger.debug("initial authorize responded but currency not found")
                    _log_trade({"type":"authorize", "token_source": "env_demo", "response_preview": auth})
                except Exception as e:
                    logger.warning("initial demo authorize failed (background): %s", e)
                    self.last_authorize_response = None
            else:
                logger.debug("no demo token configured; skipped initial authorize")
        except Exception as e:
            logger.exception("AsyncDerivClient init error: %s", e)
            self.api = None

    def run(self, coro_obj, timeout: float = 20.0):
        if not self.loop:
            logger.error("background loop not running - cannot run coroutine")
            raise RuntimeError("background loop not running")
        fut = asyncio.run_coroutine_threadsafe(coro_obj, self.loop)
        return fut.result(timeout=timeout)

    def submit(self, coro_fn, *args, timeout: float = 20.0):
        coro = coro_fn(*args)
        return self.run(coro, timeout=timeout)

    def authorize_token(self, token: str, timeout: float = 12.0):
        """
        Synchronously authorize a token in the background loop and store the response.
        Raises on failure.
        """
        if not token:
            logger.debug("authorize_token called with empty token")
            return None
        if not self.api:
            logger.warning("authorize_token: api object missing, attempting to recreate")
            try:
                self.run(self._create_api(), timeout=15.0)
            except Exception:
                logger.exception("authorize_token: failed to recreate api")
        try:
            logger.debug("authorizing token (synchronous wrapper)")
            auth = self.run(self.api.authorize(token), timeout=timeout)
            self.last_authorize_response = auth
            self.last_authorize_ts = time.time()
            cc = _extract_currency_from_auth(auth) if isinstance(auth, dict) else None
            if cc:
                self.currency = cc
                logger.info("authorize_token succeeded, currency=%s", cc)
            else:
                logger.debug("authorize_token succeeded but currency not present in response")
            _log_trade({"type":"authorize", "token_masked": (str(token)[:6] + "..."), "response_preview": auth})
            return auth
        except Exception as e:
            logger.exception("authorize_token failed: %s", e)
            _log_trade({"type":"authorize_failed", "error": str(e)})
            raise

# Instantiate background client
hd = AsyncDerivClient()

# Wait helper
def _wait_ready(timeout: float = 6.0):
    logger.debug("_wait_ready: waiting for hd._ready event")
    hd._ready.wait(timeout=timeout)
    logger.debug("_wait_ready: done wait (hd.api present=%s)", bool(hd.api))

# ---------------- Async helpers ----------------
async def _async_get_balance(api):
    try:
        if hasattr(api, "balance"):
            b = await api.balance()
            if isinstance(b, dict):
                if "balance" in b and isinstance(b["balance"], (int, float, str)):
                    return float(b["balance"])
                if "balance" in b and isinstance(b["balance"], dict) and "balance" in b["balance"]:
                    return float(b["balance"]["balance"])
            elif isinstance(b, (int, float)):
                return float(b)
    except Exception:
        pass
    try:
        a = await api.authorize(None)
        if isinstance(a, dict):
            acct = a.get("authorize", {}) or {}
            if "balance" in acct and acct.get("balance") is not None:
                return float(acct.get("balance"))
            al = acct.get("account_list", [])
            if isinstance(al, list) and len(al):
                for acc in al:
                    if acc.get("is_virtual"):
                        return float(acc.get("balance", 0))
                return float(al[0].get("balance", 0))
    except Exception:
        pass
    return None

# ---- changed helper: send proposal safely (workaround for upstream library bug) ----
async def _async_proposal(api, payload: dict):
    """
    Attempt a low-level send of the exact JSON payload over api.connection/ws and then
    wait for the 'proposal' response via api.expect_response('proposal').

    This bypasses a known bug in some python-deriv-api versions that truncate decimals
    when converting amount (e.g. 0.35 -> 0), causing minimum-stake errors.

    Behavior:
      - If api exposes a connection/ws and api.expect_response, we send a raw JSON text
        frame where amount is a TWO-DECIMAL STRING (e.g. "0.35"). Then wait for the 'proposal' msg.
      - If that fails or the api object does not expose the required internals, fall back to await api.proposal(payload).
    """
    try:
        conn = getattr(api, "connection", None) or getattr(api, "ws", None) or getattr(api, "websocket", None)
        if conn is not None and hasattr(api, "expect_response"):
            try:
                # Prepare exact JSON we'll send: make a shallow copy
                p = dict(payload)
                # Ensure amount is represented as a 2-decimal string for the manual raw send.
                if "amount" in p:
                    try:
                        dec_amt = Decimal(str(p["amount"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                        # use string for raw send
                        p["amount"] = f"{dec_amt:.2f}"
                    except Exception:
                        # On any conversion issue, leave as-is
                        pass

                # Send raw JSON as text frame directly on the websocket connection
                # Many deriv_api builds use websockets.client with an async send()
                await conn.send(json.dumps(p))

                # Wait for the 'proposal' message (library receive loop will route it)
                # Use api.expect_response to capture it (some lib versions implement this).
                # Add a small timeout safety: the caller supplies PROPOSAL_TIMEOUT; if not available, default used.
                to = PROPOSAL_TIMEOUT if isinstance(PROPOSAL_TIMEOUT, (int, float)) else 25
                resp = await asyncio.wait_for(api.expect_response("proposal"), timeout=to)
                return resp
            except Exception:
                logger.debug("_async_proposal manual send/expect failed; falling back to api.proposal", exc_info=True)
    except Exception:
        logger.debug("_async_proposal: low-level path not available", exc_info=True)

    # Final fallback: call api.proposal with the original payload
    return await api.proposal(payload)

async def _async_buy(api, buy_req: dict):
    return await api.buy(buy_req)

async def _async_proposal_open_contract(api, contract_id: Any):
    return await api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": contract_id})

async def _async_close_api(api):
    if not api:
        return
    try:
        close_m = getattr(api, "close", None)
        if close_m:
            maybe = close_m()
            if asyncio.iscoroutine(maybe):
                await maybe
            return
    except Exception:
        pass
    try:
        conn = getattr(api, "connection", None) or getattr(api, "ws", None) or getattr(api, "websocket", None)
        if conn:
            close_m = getattr(conn, "close", None)
            if close_m:
                maybe = close_m()
                if asyncio.iscoroutine(maybe):
                    await maybe
    except Exception:
        pass

# ---------------- Poll contract settlement (async) ----------------
async def _async_poll_settlement(api, contract_id: Any, timeout_seconds: float = SETTLEMENT_POLL_TIMEOUT, poll_interval: float = SETTLEMENT_POLL_INTERVAL):
    start = time.time()
    last = None
    while True:
        try:
            resp = await _async_proposal_open_contract(api, contract_id)
            last = resp
            candidate = resp.get("proposal_open_contract") or resp.get("contract") or resp
            if isinstance(candidate, dict):
                contract = None
                if "contract" in candidate and isinstance(candidate["contract"], dict):
                    contract = candidate["contract"]
                elif isinstance(candidate, dict) and any(k in candidate for k in ("is_sold", "is_settled", "status", "sell_price", "profit")):
                    contract = candidate
                if isinstance(contract, dict):
                    if contract.get("is_settled") is True or contract.get("is_sold") is True:
                        return True, resp
                    st = (contract.get("status") or "").lower()
                    if st in ("sold", "settled", "closed"):
                        return True, resp
                    if "sell_price" in contract or "profit" in contract or "payout" in contract:
                        if contract.get("sell_price") is not None:
                            return True, resp
        except Exception:
            pass
        if time.time() - start > timeout_seconds:
            return False, last
        await asyncio.sleep(poll_interval)

# ---------------- Reconnect helpers ----------------
def _is_ws_or_conn_error(exc: Exception) -> bool:
    if exc is None:
        return False
    try:
        import websockets as _ws_mod
        if isinstance(exc, _ws_mod.exceptions.ConnectionClosedError) or isinstance(exc, _ws_mod.exceptions.ConnectionClosed):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    checks = [
        "no close frame", "connectionclosederror", "connection reset by peer",
        "incomplete read", "remote host closed", "connection was closed",
        "connection aborted", "broken pipe", "eof", "cancellederror", "timeout"
    ]
    for c in checks:
        if c in msg:
            return True
    return False

def _proposal_with_reconnect(payload: dict, retries: int = 1):
    attempt = 0
    last_exc = None
    while True:
        attempt += 1
        try:
            logger.debug("proposal attempt %s payload=%s", attempt, payload)
            # use a shallow copy for sending to avoid any library-side mutation of our dict
            payload_to_send = dict(payload)

            # Create a send-copy that forces amount -> two-decimal STRING (e.g. "0.35")
            payload_for_send = dict(payload_to_send)
            if "amount" in payload_for_send:
                try:
                    dec_amt = Decimal(str(payload_for_send.get("amount"))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                    payload_for_send["amount"] = f"{dec_amt:.2f}"
                except Exception:
                    # if conversion fails, leave original
                    pass

            # debug: exact JSON being sent (this will appear in your logs)
            try:
                logger.debug("DERIV PROPOSAL (pre-send) attempt=%s original_amount=%r original_type=%s send_amount=%r send_type=%s json=%s",
                             attempt,
                             payload_to_send.get("amount"),
                             type(payload_to_send.get("amount")),
                             payload_for_send.get("amount"),
                             type(payload_for_send.get("amount")),
                             json.dumps(payload_for_send, default=str))
            except Exception:
                logger.debug("DERIV PROPOSAL (pre-send) attempt=%s (json dump failed)", attempt)

            # send the payload_for_send (amount as string) so both raw send and fallback get the string amount
            resp = hd.submit(_async_proposal, hd.api, payload_for_send, timeout=PROPOSAL_TIMEOUT)
            logger.debug("proposal response (attempt %s): %s", attempt, resp)
            _log_trade({"type":"proposal_response", "attempt": attempt, "payload": payload_for_send, "response_preview": (resp if isinstance(resp, dict) else str(resp))})
            return resp
        except Exception as e:
            last_exc = e
            is_ws_err = _is_ws_or_conn_error(e)
            is_timeout = isinstance(e, (concurrent.futures.TimeoutError, asyncio.TimeoutError, TimeoutError))
            logger.warning("proposal attempt %s failed: %s (ws_err=%s timeout=%s)", attempt, e, is_ws_err, is_timeout)
            if attempt > retries + 1 or (not is_ws_err and not is_timeout):
                logger.exception("proposal final failure")
                raise
            try:
                logger.info("proposal reconnect attempt: closing and recreating api")
                try:
                    hd.run(_async_close_api(hd.api), timeout=6.0)
                except Exception:
                    logger.exception("error closing api during reconnect")
                try:
                    hd.run(hd._create_api(), timeout=15.0)
                except Exception as e2:
                    logger.exception("recreate hd.api failed: %s", e2)
                    raise e2 from e
                time.sleep(0.25)
                continue
            except Exception:
                logger.exception("reconnect flow failed")
                raise

def _buy_with_reconnect(buy_req: dict, retries: int = 1):
    attempt = 0
    last_exc = None
    while True:
        attempt += 1
        try:
            logger.debug("buy attempt %s buy_req=%s", attempt, buy_req)
            resp = hd.submit(_async_buy, hd.api, buy_req, timeout=BUY_TIMEOUT)
            logger.debug("buy response (attempt %s): %s", attempt, resp)
            _log_trade({"type":"buy_response", "attempt": attempt, "buy_req": buy_req, "response_preview": (resp if isinstance(resp, dict) else str(resp))})
            return resp
        except Exception as e:
            last_exc = e
            is_ws_err = _is_ws_or_conn_error(e)
            is_timeout = isinstance(e, (concurrent.futures.TimeoutError, asyncio.TimeoutError, TimeoutError))
            logger.warning("buy attempt %s failed: %s (ws_err=%s timeout=%s)", attempt, e, is_ws_err, is_timeout)
            if attempt > retries + 1 or (not is_ws_err and not is_timeout):
                logger.exception("buy final failure")
                raise
            try:
                logger.info("buy reconnect attempt: closing and recreating api")
                try:
                    hd.run(_async_close_api(hd.api), timeout=6.0)
                except Exception:
                    logger.exception("error closing api during buy reconnect")
                try:
                    hd.run(hd._create_api(), timeout=15.0)
                except Exception as e2:
                    logger.exception("recreate hd.api failed (buy): %s", e2)
                    raise e2 from e
                time.sleep(0.25)
                continue
            except Exception:
                logger.exception("buy reconnect flow failed")
                raise

# ---------------- Core trade execution helper ----------------
def _execute_trade(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Central trade logic: proposal -> buy -> poll settlement
    All important events are logged via _log_trade and logger.
    """
    trace_id = f"trace_{int(time.time() * 1000)}"
    _log_trade({"type":"incoming_execute_trade", "trace_id": trace_id, "raw_body": body})
    logger.info("execute_trade start trace=%s body=%s", trace_id, body)

    symbol = (body.get("symbol") or "").strip().upper()
    direction_in = body.get("direction") or body.get("signal") or "higher"
    stake = body.get("stake", DEFAULT_STAKE)
    duration = int(body.get("duration", 5))
    mode = (body.get("mode") or "demo").lower()
    force_take = bool(body.get("force", False) or body.get("force_trade", False))

    if not symbol:
        logger.warning("execute_trade missing symbol")
        return {"ok": False, "error": "missing_symbol"}

    try:
        contract_type = _normalize_direction(direction_in)
    except Exception as e:
        logger.warning("execute_trade invalid direction: %s", e)
        return {"ok": False, "error": "invalid_direction", "detail": str(e)}

    # normalize stake
    try:
        stake = float(stake)
    except Exception:
        stake = float(DEFAULT_STAKE)
    stake = round(stake + 1e-9, 2)
    if stake < MIN_STAKE:
        stake = MIN_STAKE

    # determine barrier default based on direction (unify with front-end)
    barrier_from_body = body.get("barrier")
    if barrier_from_body is None or barrier_from_body == "":
        # keep same convention as the UI: CALL (higher) -> "-1", PUT (lower) -> "+1"
        barrier = "-1" if contract_type == "CALL" else "+1"
    else:
        barrier = barrier_from_body

    # ensure background client ready
    _wait_ready(timeout=8.0)
    if not hd.api:
        logger.warning("execute_trade: hd.api not ready, attempting to recreate")
        try:
            hd.run(hd._create_api(), timeout=15.0)
        except Exception:
            logger.exception("execute_trade: failed to recreate hd.api")
    if not hd.api:
        logger.error("execute_trade: hd.api still not ready")
        _log_trade({"type":"trade_aborted", "reason":"deriv_api_not_ready", "trace_id": trace_id})
        return {"ok": False, "error": "deriv_api_not_ready"}

    # select token source: prefer env tokens if present, otherwise ask hero_service for server token
    token, token_source = resolve_token(mode)
    logger.debug("execute_trade selected mode=%s token_source=%s", mode, token_source)

    if token:
        try:
            try:
                auth_resp = hd.authorize_token(token)
                logger.debug("authorize result preview: %s", (auth_resp if isinstance(auth_resp, dict) else str(auth_resp)))
                if isinstance(auth_resp, dict):
                    cc = _extract_currency_from_auth(auth_resp)
                    if cc:
                        hd.currency = cc
                        logger.info("execute_trade: updated hd.currency=%s", cc)
            except Exception as e:
                logger.warning("authorize_token encountered error: %s", e)
        except Exception:
            logger.exception("execute_trade authorize fallback (ignored)")

    currency = hd.currency or DEFAULT_CURRENCY or "USD"
    payload = _build_proposal_payload(symbol, contract_type, stake, barrier, duration, currency=currency)

    logger.info("execute_trade proposal payload: %s", payload)
    _log_trade({"type":"trade_proposal_attempt", "trace_id": trace_id, "payload": payload, "mode": mode, "hd_currency": hd.currency, "token_source": token_source})

    # get balance before (best-effort)
    try:
        balance_before = hd.submit(_async_get_balance, hd.api, timeout=8.0)
        logger.debug("balance_before: %s", balance_before)
    except Exception as e:
        balance_before = None
        logger.warning("failed to get balance_before: %s", e)

    # call proposal
    try:
        proposal = _proposal_with_reconnect(payload, retries=1)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("proposal failed: %s", e)
        _log_trade({"type":"trade_proposal_failed", "trace_id": trace_id, "payload":payload, "error": str(e), "trace": tb})
        return {"ok": False, "error": "proposal_failed", "detail": str(e), "trace": tb}

    if isinstance(proposal, dict) and proposal.get("error"):
        logger.warning("proposal returned error: %s", proposal.get("error"))
        _log_trade({"type":"trade_proposal_error", "trace_id": trace_id, "payload":payload, "proposal":proposal})
        return {"ok": False, "error": "proposal_error", "proposal": proposal}

    pid = _extract_proposal_id(proposal)
    ask_price = _extract_ask_price(proposal)
    if ask_price is None:
        ask_price = stake

    _log_trade({"type":"proposal_ok", "trace_id": trace_id, "pid": pid, "ask_price": ask_price, "proposal_preview": proposal})
    logger.info("proposal ok pid=%s ask_price=%s", pid, ask_price)

    if not pid:
        logger.error("execute_trade: proposal id missing")
        _log_trade({"type":"trade_no_proposal_id", "trace_id": trace_id, "payload":payload, "proposal":proposal})
        return {"ok": False, "error": "proposal_id_missing", "proposal": proposal}

    buy_req = {"buy": pid}
    try:
        buy_req["price"] = float(ask_price)
    except Exception:
        buy_req["price"] = ask_price

    if not force_take:
        logger.debug("execute_trade: not forced (front-end may enforce), proceeding since this helper is permissive")
    else:
        logger.debug("execute_trade: force_take=True")

    # execute buy
    try:
        buy_resp = _buy_with_reconnect(buy_req, retries=1)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("buy failed: %s", e)
        _log_trade({"type":"trade_buy_failed", "trace_id": trace_id, "payload":payload, "proposal":proposal, "error": str(e), "trace": tb})
        return {"ok": False, "error": "buy_failed", "detail": str(e), "trace": tb}

    if isinstance(buy_resp, dict) and buy_resp.get("error"):
        logger.warning("buy returned error body: %s", buy_resp.get("error"))
        _log_trade({"type":"trade_buy_error", "trace_id": trace_id, "payload":payload, "proposal":proposal, "buy_resp":buy_resp})
        return {"ok": False, "error": "buy_error", "buy": buy_resp, "proposal": proposal}

    contract_id = None
    try:
        contract_id = buy_resp.get("buy", {}).get("contract_id") or buy_resp.get("contract_id")
    except Exception:
        contract_id = None

    _log_trade({"type":"trade_bought", "trace_id": trace_id, "buy_resp": buy_resp, "contract_id": contract_id})
    logger.info("buy response received contract_id=%s", contract_id)

    settled = False
    contract_status = None
    if contract_id:
        try:
            settled, contract_status = hd.submit(_async_poll_settlement, hd.api, contract_id, SETTLEMENT_POLL_TIMEOUT)
            logger.info("poll_settlement returned settled=%s", settled)
            _log_trade({"type":"settlement_poll", "trace_id": trace_id, "contract_id": contract_id, "settled": settled, "status_preview": contract_status})
        except Exception as e:
            logger.exception("settlement poll failed: %s", e)
            try:
                settled, contract_status = hd.submit(_async_poll_settlement, hd.api, contract_id, duration * 6 + 10.0)
                _log_trade({"type":"settlement_poll_fallback", "trace_id": trace_id, "contract_id": contract_id, "settled": settled, "status_preview": contract_status})
            except Exception:
                settled = False
                contract_status = None

    try:
        balance_after = hd.submit(_async_get_balance, hd.api, timeout=8.0)
        logger.debug("balance_after: %s", balance_after)
    except Exception as e:
        balance_after = None
        logger.warning("failed to get balance_after: %s", e)

    # --- Minimal change: result only "win" or "loss" ---
    profit = None
    result = "loss"
    try:
        if balance_before is not None and balance_after is not None:
            profit = round(float(balance_after) - float(balance_before), 8)
            result = "win" if profit > 0 else "loss"
    except Exception:
        profit = None
        result = "loss"

    out = {
        "ok": True,
        "trace_id": trace_id,
        "proposal": proposal,
        "buy": buy_resp,
        "contract_id": contract_id,
        "settled": bool(settled),
        "contract_status": contract_status,
        "balance_before": balance_before,
        "balance_after": balance_after,
        "profit": profit,
        "result": result,
    }

    _log_trade({"type":"trade_result", "trace_id": trace_id, "mode": mode, "payload": payload, "proposal": proposal, "buy": buy_resp, "contract_id": contract_id, "settled": settled, "contract_status": contract_status, "balance_before": balance_before, "balance_after": balance_after, "profit": profit, "result": result})
    logger.info("execute_trade finished trace=%s result=%s profit=%s", trace_id, result, profit)
    return out

# ---------------- Flask endpoints ----------------
@app.route("/simulate", methods=["POST"])
def simulate():
    """
    Simulate a proposal using ticks.
    Body JSON:
      { "symbol": "RDBEAR", "direction": "higher", "stake": 1.0, "barrier": "+1", "duration": 5, "mode":"demo" }
    Duration is in ticks.
    """
    try:
        body = request.get_json(force=True)
    except Exception:
        logger.warning("/simulate invalid_json")
        return jsonify({"ok": False, "error": "invalid_json"}), 400

    symbol = (body.get("symbol") or "").strip().upper()
    direction_in = body.get("direction") or "higher"
    stake = body.get("stake", DEFAULT_STAKE)
    duration = int(body.get("duration", 5))
    mode = (body.get("mode") or "demo").lower()

    if not symbol:
        logger.warning("/simulate missing_symbol")
        return jsonify({"ok": False, "error": "missing_symbol"}), 400
    try:
        contract_type = _normalize_direction(direction_in)
    except Exception as e:
        logger.warning("/simulate invalid direction %s", e)
        return jsonify({"ok": False, "error": str(e)}), 400

    try:
        stake = float(stake)
    except Exception:
        stake = float(DEFAULT_STAKE)
    stake = round(stake + 1e-9, 2)
    if stake < MIN_STAKE:
        stake = MIN_STAKE

    # barrier default aligned with direction
    barrier_from_body = body.get("barrier")
    barrier = barrier_from_body if (barrier_from_body is not None and barrier_from_body != "") else ("-1" if contract_type == "CALL" else "+1")

    _wait_ready(timeout=8.0)
    if not hd.api:
        logger.error("/simulate deriv_api_not_ready")
        return jsonify({"ok": False, "error": "deriv_api_not_ready"}), 500

    token = DEFAULT_TOKEN_REAL if mode == "real" and DEFAULT_TOKEN_REAL else DEFAULT_TOKEN_DEMO
    currency = hd.currency or DEFAULT_CURRENCY or "USD"
    payload = _build_proposal_payload(symbol, contract_type, stake, barrier, duration, currency=currency)

    logger.info("/simulate calling proposal payload=%s mode=%s currency=%s", payload, mode, currency)
    _log_trade({"type":"simulate_attempt", "payload": payload, "mode": mode})

    try:
        prop = _proposal_with_reconnect(payload, retries=1)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("/simulate proposal_failed %s", e)
        _log_trade({"type":"simulate_failed", "payload":payload, "error": str(e), "trace": tb})
        return jsonify({"ok": False, "error": "proposal_failed", "detail": str(e), "trace": tb}), 500

    _log_trade({"type":"simulate", "mode": mode, "payload": payload, "proposal": prop})
    return jsonify({"ok": True, "proposal_request": payload, "proposal": prop}), 200

@app.route("/trade", methods=["POST"])
def trade():
    """
    Place a trade: get proposal then buy it and poll for settlement.
    Delegates to _execute_trade to centralize logic and logging.
    """
    try:
        body = request.get_json(force=True)
    except Exception:
        logger.warning("/trade invalid_json")
        return jsonify({"ok": False, "error": "invalid_json"}), 400

    # pass-through logging for correlation
    try:
        _log_trade({"type":"incoming_trade_endpoint", "raw_body": body if isinstance(body, dict) else str(body)})
    except Exception:
        logger.exception("failed to log incoming trade forward")

    res = _execute_trade(body)
    status_code = 200 if res.get("ok") else 400
    return jsonify(res), status_code

@app.route("/signal", methods=["POST"])
def signal():
    """
    Accept a simple signal payload and attempt to place a trade.
    Example:
      { "signal": "higher", "symbol": "RDBEAR", "force": true, "mode": "demo" }
    """
    try:
        body = request.get_json(force=True)
    except Exception:
        logger.warning("/signal invalid_json")
        return jsonify({"ok": False, "error": "invalid_json"}), 400

    # normalize: allow "higher"/"lower" or "take_higher"/"take_lower"
    sig = (body.get("signal") or body.get("action") or "").strip().lower()
    if not sig:
        logger.warning("/signal missing signal")
        return jsonify({"ok": False, "error": "missing_signal"}), 400

    if sig.startswith("take_"):
        sig = sig.split("take_")[-1]

    if sig not in ("higher", "lower", "call", "put", "up", "down"):
        logger.warning("/signal invalid signal %s", sig)
        return jsonify({"ok": False, "error": "invalid_signal"}), 400

    # determine default barrier consistently
    normalized = "higher" if sig in ("higher","call","up") else "lower"
    default_barrier = "-1" if normalized == "higher" else "+1"
    payload = {
        "symbol": (body.get("symbol") or "RDBEAR"),
        "signal": sig,
        "direction": sig,
        "stake": body.get("stake", DEFAULT_STAKE),
        "barrier": body.get("barrier") if (body.get("barrier") is not None and body.get("barrier") != "") else default_barrier,
        "duration": int(body.get("duration", 5)),
        "mode": (body.get("mode") or "demo"),
        "force": bool(body.get("force", True) or body.get("force_trade", False)),
    }

    # DEBUG: log incoming stake shape/type so we can verify frontend->server value
    try:
        logger.debug("/signal received raw stake=%r type=%s body_stake=%r", body.get("stake"), type(body.get("stake")), payload.get("stake"))
    except Exception:
        logger.debug("/signal received (debug stake log failed)")

    _log_trade({"type":"incoming_signal", "signal": sig, "payload": payload})
    res = _execute_trade(payload)
    status_code = 200 if res.get("ok") else 400
    return jsonify(res), status_code

# ----- Blueprint shim so hero_service can register HL routes -----
hl_bp = _Blueprint("higher_lower_trade", __name__, url_prefix="/trader")

@hl_bp.route("/simulate", methods=["POST"])
def _hl_simulate_proxy():
    return simulate()

@hl_bp.route("/trade", methods=["POST"])
def _hl_trade_proxy():
    return trade()

@hl_bp.route("/signal", methods=["POST"])
def _hl_signal_proxy():
    return signal()

@hl_bp.route("/status", methods=["GET"])
def _hl_status_proxy():
    return status()

# ---------------- Status / health ----------------
@app.route("/status", methods=["GET"])
def status():
    """
    Return simple health/status info about hd background client and tokens.
    """
    try:
        _wait_ready(timeout=1.0)
        status = {
            "hd_ready": bool(hd._ready.is_set()),
            "hd_loop": bool(hd.loop is not None),
            "hd_thread_alive": bool(hd.thread and hd.thread.is_alive()),
            "hd_api_present": bool(hd.api is not None),
            "hd_currency": hd.currency,
            "hd_last_authorize_ts": hd.last_authorize_ts,
            "demo_token_present": bool(DEFAULT_TOKEN_DEMO),
            "real_token_present": bool(DEFAULT_TOKEN_REAL),
        }
        logger.debug("/status requested: %s", status)
        return jsonify({"ok": True, "status": status}), 200
    except Exception as e:
        logger.exception("/status failed")
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    logger.info("Starting higher_lower_trade (verbose logging enabled)")
    logger.info("No hard-coded token will be used. Provide tokens via DERIV_TOKEN (demo) or DERIV_TOKEN_REAL (real) or via hero_service control endpoints.")
    time.sleep(0.25)
    # use a different standalone port to avoid colliding with hero_service which usually runs on 5000
    app.run(host="127.0.0.1", port=int(os.environ.get("HL_STANDALONE_PORT", "5001")))