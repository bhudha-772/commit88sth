#!/usr/bin/env python3
from __future__ import annotations
"""
deriv_worker_async.py

- Connects to Deriv websocket, subscribes to ticks for configured SYMBOLS
- Appends ticks to ticks.csv
- Optionally posts each tick JSON to HERO_DASHBOARD_PUSH_URL (so dashboard broadcasts via SSE)
- Periodically aggregates ticks to 1-minute bars and calls TimeSeriesPipeline.train_and_backtest()
- Fallback to raw websockets if deriv_api is not installed.

This variant adds a trailing-zero inference heuristic:
- keeps a short per-symbol history of observed decimal-digit length,
- infers the most common length and pads incoming ticks that are shorter to that length.
Configure via HERO_INFER_TRAILING_ZEROS env (default 1) or CLI --no-infer-zero.
"""

import argparse
import asyncio
import csv
import os
import time
import threading
import json
import signal
from collections import deque, Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Deque, List, Optional, Dict

# third-party optional
try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

# websockets and requests are required (requests used for dashboard POST)
import websockets
import requests

# try to import deriv_api (optional)
try:
    from deriv_api import DerivAPI  # type: ignore
    DERIV_API_AVAILABLE = True
except Exception:
    DerivAPI = None  # type: ignore
    DERIV_API_AVAILABLE = False

# your pipeline implementation (if present). If missing, pipeline steps are skipped.
try:
    from timeseries_pipeline import TimeSeriesPipeline  # type: ignore
    PIPELINE_AVAILABLE = True
except Exception:
    TimeSeriesPipeline = None  # type: ignore
    PIPELINE_AVAILABLE = False

# ---------- CONFIG (defaults, override via env or CLI) ----------
DEFAULT_APP_ID = int(os.environ.get("DERIV_APP_ID", os.environ.get("HERO_DERIV_APP_ID", "71710")))
DERIV_WS_URL = os.environ.get("DERIV_WS_URL", f"wss://ws.binaryws.com/websockets/v3?app_id={DEFAULT_APP_ID}")
# Allow alternative official endpoint naming from older codebases:
if "derivws.com" in DERIV_WS_URL and "binaryws.com" not in DERIV_WS_URL:
    pass

# symbols default (can be overridden via HERO_SYMBOLS env or --symbols)
DEFAULT_SYMBOLS = ["R_100"]
ENV_SYMBOLS = os.environ.get("HERO_SYMBOLS") or os.environ.get("SYMBOLS")
if ENV_SYMBOLS:
    try:
        DEFAULT_SYMBOLS = [s.strip().upper() for s in ENV_SYMBOLS.split(",") if s.strip()]
    except Exception:
        DEFAULT_SYMBOLS = DEFAULT_SYMBOLS

BUFFER_MAXLEN = int(os.environ.get("HERO_BUFFER_MAXLEN", "20000"))
TICKS_CSV = os.environ.get("HERO_TICKS_CSV", "ticks.csv")
AGGREGATE_INTERVAL_SECONDS = int(os.environ.get("HERO_AGG_INTERVAL", "60"))
BAR_FREQ = os.environ.get("HERO_BAR_FREQ", "1T")
BACKTEST_PERIOD_BARS = int(os.environ.get("HERO_BACKTEST_BARS", "800"))
RUN_PIPELINE_ON_AGG = (os.environ.get("HERO_RUN_PIPELINE", "1") == "1") and PANDAS_AVAILABLE and PIPELINE_AVAILABLE

# Dashboard push URL (set by dashboard when spawning worker or via env)
DASH_PUSH_URL = os.environ.get("HERO_DASHBOARD_PUSH_URL", "").strip() or None

# Trailing-zero inference: default enabled (1). Set HERO_INFER_TRAILING_ZEROS=0 to disable.
INFER_TRAILING_ZEROS_ENV = os.environ.get("HERO_INFER_TRAILING_ZEROS", "1")
DEFAULT_INFER_TRAILING_ZEROS = INFER_TRAILING_ZEROS_ENV not in ("0", "false", "False", "")

# ensure CSV exists with header
if not os.path.exists(TICKS_CSV):
    try:
        with open(TICKS_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "epoch",
                    "price_raw",
                    "price_str_text",
                    "price",
                    "last_decimal_digit",
                    "last_unit_digit",
                    "symbol",
                ]
            )
    except Exception:
        pass


def _safe_post_json(url: str, json_payload: dict, timeout: float = 2.0) -> None:
    """Fire-and-forget POST in a background thread to avoid blocking the loop."""
    def _do():
        try:
            requests.post(url, json=json_payload, timeout=timeout)
        except Exception:
            pass
    threading.Thread(target=_do, daemon=True).start()


class DerivWorker:
    def __init__(self, symbols: List[str], infer_trailing_zeros: bool = DEFAULT_INFER_TRAILING_ZEROS):
        self.symbols = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
        self.buffer: Deque[tuple[int, str, float, str, str, str]] = deque(maxlen=BUFFER_MAXLEN)
        self.api = None
        self._raw_conn = None  # websocket.WebSocketClientProtocol
        self.pipeline = TimeSeriesPipeline() if (PIPELINE_AVAILABLE and RUN_PIPELINE_ON_AGG) else None
        self._stop_event = asyncio.Event()
        self._recv_task: Optional[asyncio.Task] = None
        self._aggregator_task: Optional[asyncio.Task] = None
        self._using_deriv_api = False
        self._subscriptions = set()

        # trailing-zero inference controls
        self.infer_trailing_zeros = bool(infer_trailing_zeros)
        # per-symbol rolling history (deque) of observed decimal lengths
        self.decimal_history: Dict[str, deque] = {}

        # how many history items to keep per symbol
        self._hist_maxlen = 20

    # ---------- Price string preservation helpers ----------
    @staticmethod
    def _format_price_string_preserve(price_raw) -> str:
        """
        Convert price_raw to a cleaned string while preserving textual precision when possible.

        Prefer the exact string sent by server. If numeric, convert using Decimal(str(value))
        to avoid floating artifacts. Do not force any quantization here.
        """
        if price_raw is None:
            return ""
        if isinstance(price_raw, str):
            s = price_raw.strip()
        else:
            try:
                s = format(Decimal(str(price_raw)), 'f')
            except Exception:
                s = str(price_raw)
        return s.replace(",", "")

    @staticmethod
    def _count_decimal_len_from_string(s: str) -> int:
        if not s:
            return 0
        s = s.strip()
        if '.' in s:
            _, dec = s.split('.', 1)
            # consider only digits in decimal part
            dec_digits = ''.join(ch for ch in dec if ch.isdigit())
            return len(dec_digits)
        return 0

    def _preferred_decimal_length(self, symbol: str) -> int:
        hist = self.decimal_history.get(symbol)
        if not hist:
            return 0
        cnt = Counter(hist)
        # most_common returns list of (value, count)
        mc = cnt.most_common(1)
        return int(mc[0][0]) if mc else 0

    def _maybe_update_decimal_history(self, symbol: str, observed_len: int) -> None:
        if symbol not in self.decimal_history:
            self.decimal_history[symbol] = deque(maxlen=self._hist_maxlen)
        self.decimal_history[symbol].append(int(observed_len))

    def _pad_price_string_to_len(self, price_str: str, target_len: int) -> str:
        if target_len <= 0:
            return price_str
        s = price_str or ""
        if "." in s:
            intp, decp = s.split(".", 1)
            dec_digits = ''.join(ch for ch in decp if ch.isdigit())
            need = max(0, target_len - len(dec_digits))
            decp = dec_digits + ("0" * need)
            return f"{intp}.{decp}"
        else:
            # no decimal part present, create one of target_len zeros
            return f"{s or '0'}.{('0'*target_len)}"

    def _extract_last_digits(self, price_str: str) -> tuple[str, str]:
        """
        Extract last decimal digit and last unit digit from the (possibly padded) price string.
        """
        if not price_str:
            return "0", "0"
        s = price_str.strip()
        if "." in s:
            integer_part, decimal_part = s.split(".", 1)
        else:
            integer_part, decimal_part = s, ""
        # preserve digits only
        dec_digits = ''.join(ch for ch in decimal_part if ch.isdigit())
        last_decimal_digit = dec_digits[-1] if dec_digits else "0"
        integer_clean = ''.join(ch for ch in integer_part if ch.isdigit())
        last_unit_digit = integer_clean[-1] if integer_clean else "0"
        return last_decimal_digit, last_unit_digit

    def _process_and_store(self, epoch: int, price_raw, price_str_text: str, price_num: float, last_decimal: str, last_unit: str, symbol: str):
        # Append to CSV
        try:
            with open(TICKS_CSV, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch, str(price_raw), price_str_text, price_num, last_decimal, last_unit, symbol])
        except Exception:
            pass

        # Push to dashboard (if URL set)
        if DASH_PUSH_URL:
            try:
                payload = {
                    "epoch": int(epoch),
                    "price_raw": str(price_raw),
                    "price_str_text": price_str_text,
                    "price": float(price_num) if price_num is not None else None,
                    "last_decimal": last_decimal,
                    "last_unit": last_unit,
                    "symbol": symbol,
                    "ts": time.time(),
                }
                _safe_post_json(DASH_PUSH_URL, payload)
            except Exception:
                pass

    async def on_tick(self, tick_event):
        try:
            # tick_event expected shape containing 'tick' key or be already the tick dict
            t = tick_event.get("tick") if isinstance(tick_event, dict) and "tick" in tick_event else tick_event
            if not isinstance(t, dict):
                return

            epoch_raw = t.get("epoch") or t.get("epoch_l") or int(time.time())
            try:
                epoch = int(epoch_raw)
            except Exception:
                epoch = int(float(epoch_raw) if epoch_raw else time.time())

            # Prefer quote, then price, then ask/bid
            price_raw = None
            if t.get("quote") is not None:
                price_raw = t.get("quote")
            elif t.get("price") is not None:
                price_raw = t.get("price")
            elif t.get("ask") is not None:
                price_raw = t.get("ask")
            elif t.get("bid") is not None:
                price_raw = t.get("bid")
            else:
                for k in ("quote", "price", "ask", "bid"):
                    if k in t:
                        price_raw = t.get(k)
                        break

            # symbol determination
            symbol = None
            if isinstance(t, dict):
                symbol = t.get("symbol")
            if not symbol and isinstance(tick_event, dict):
                try:
                    symbol = tick_event.get("echo_req", {}).get("ticks")
                except Exception:
                    symbol = None
            if not symbol:
                symbol = self.symbols[0] if self.symbols else "R_100"
            symbol = symbol.upper() if isinstance(symbol, str) else str(symbol)

            # Produce preserved string
            price_str_original = self._format_price_string_preserve(price_raw)

            # observed decimal length (as provided or as produced)
            observed_len = self._count_decimal_len_from_string(price_str_original)

            # update history
            self._maybe_update_decimal_history(symbol, observed_len)

            # decide whether to pad using preferred length
            price_str_to_use = price_str_original
            if self.infer_trailing_zeros:
                preferred_len = self._preferred_decimal_length(symbol)
                # if preferred_len > observed_len then pad with zeros
                if preferred_len > observed_len:
                    price_str_to_use = self._pad_price_string_to_len(price_str_original, preferred_len)

            # Convert to numeric using Decimal from the (possibly padded) string
            try:
                price_num = float(Decimal(price_str_to_use)) if price_str_to_use != "" else float("nan")
            except Exception:
                try:
                    price_num = float(str(price_raw))
                except Exception:
                    price_num = float("nan")

            # last digits from the potentially padded string
            last_decimal_digit, last_unit_digit = self._extract_last_digits(price_str_to_use)

            # buffer append (epoch, price_str, price_num, last_decimal, last_unit, symbol)
            self.buffer.append((epoch, price_str_to_use, price_num, last_decimal_digit, last_unit_digit, symbol))

            # prefix for excel display
            price_str_text = "'" + price_str_to_use

            # store and push
            self._process_and_store(epoch, price_raw, price_str_text, price_num, last_decimal_digit, last_unit_digit, symbol)

            ts = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[tick] {symbol} @ {ts} => {price_str_to_use}  (last_dec:{last_decimal_digit}, last_unit:{last_unit_digit})")
        except Exception as e:
            print("on_tick error:", repr(e))

    # ---------- subscription implementations ----------
    async def subscribe_with_deriv_api(self):
        """
        Use deriv_api (if available). The library exposes a subscribe() helper that returns
        an observable-like object we can attach callbacks to.
        """
        if not DERIV_API_AVAILABLE:
            raise RuntimeError("deriv_api not available")
        print("Connecting using deriv_api to:", DERIV_WS_URL)
        try:
            self._raw_conn = await websockets.connect(DERIV_WS_URL, max_size=None)
        except Exception as e:
            raise RuntimeError(f"websocket connect failed: {e}")

        try:
            self.api = DerivAPI(connection=self._raw_conn)
            self._using_deriv_api = True
        except Exception:
            self.api = DerivAPI(app_id=DEFAULT_APP_ID)
            self._using_deriv_api = True

        for s in self.symbols:
            try:
                src = await self.api.subscribe({"ticks": s, "subscribe": 1})
                try:
                    src.subscribe(lambda ev, _self=self: asyncio.create_task(_self.on_tick(ev)))
                    print(f"Subscribed (deriv_api) to {s}")
                except Exception:
                    print(f"Subscribed (deriv_api) to {s} (fallback, callback attach failed)")
            except Exception as e:
                print(f"subscribe error for {s}: {e}")
        return

    async def subscribe_with_raw_ws(self):
        """
        Raw websockets approach. Send subscribe messages for each symbol and receive loop
        will dispatch tick messages.
        """
        print("Connecting raw websocket to:", DERIV_WS_URL)
        try:
            self._raw_conn = await websockets.connect(DERIV_WS_URL, max_size=None)
        except Exception as e:
            raise RuntimeError(f"websocket connect failed: {e}")

        for s in self.symbols:
            try:
                req = {"ticks": s, "subscribe": 1}
                await self._raw_conn.send(json.dumps(req))
                self._subscriptions.add(s)
                print(f"Raw subscribe sent for {s}")
            except Exception as e:
                print(f"raw subscribe send error for {s}: {e}")

        async def _recv_loop():
            try:
                while not self._stop_event.is_set():
                    try:
                        msg = await asyncio.wait_for(self._raw_conn.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.ConnectionClosed:
                        print("raw ws connection closed")
                        break
                    try:
                        if isinstance(msg, bytes):
                            try:
                                msg = msg.decode("utf-8")
                            except Exception:
                                continue
                        data = json.loads(msg)
                    except Exception:
                        continue
                    if isinstance(data, dict) and "tick" in data:
                        try:
                            await self.on_tick(data)
                        except Exception:
                            pass
            except Exception as e:
                print("raw recv loop error:", repr(e))
            finally:
                try:
                    await self._raw_conn.close()
                except Exception:
                    pass

        self._recv_task = asyncio.create_task(_recv_loop())

    async def subscribe_all(self):
        if DERIV_API_AVAILABLE:
            try:
                await self.subscribe_with_deriv_api()
                return
            except Exception as e:
                print("deriv_api attempt failed, falling back to raw websockets:", e)
        await self.subscribe_with_raw_ws()

    # ---------- aggregator ----------
    def buffer_to_dataframe(self):
        if not PANDAS_AVAILABLE:
            raise RuntimeError("pandas not available")
        if not self.buffer:
            return pd.DataFrame(columns=["price", "price_str", "symbol", "last_decimal_digit", "last_unit_digit"])
        rows = list(self.buffer)
        df = pd.DataFrame(rows, columns=["epoch", "price_str", "price", "last_decimal_digit", "last_unit_digit", "symbol"])
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
        df = df.set_index("datetime").sort_index()
        return df[["price", "price_str", "symbol", "last_decimal_digit", "last_unit_digit"]]

    async def aggregator_loop(self):
        while not self._stop_event.is_set():
            try:
                if len(self.buffer) == 0:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=AGGREGATE_INTERVAL_SECONDS)
                    continue
                if not PANDAS_AVAILABLE or not PIPELINE_AVAILABLE or not RUN_PIPELINE_ON_AGG:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=AGGREGATE_INTERVAL_SECONDS)
                    continue
                df_ticks = self.buffer_to_dataframe()
                bars = df_ticks["price"].resample(BAR_FREQ).last().ffill().dropna().to_frame()
                bars.columns = ["price"]
                print(f"[agg] OHLC bars length: {len(bars)}; buffer len: {len(self.buffer)}")
                try:
                    bars.tail(5).to_csv("latest_bars_head.csv")
                except Exception:
                    pass
                try:
                    bars.to_parquet("bars_latest.parquet")
                except Exception:
                    pass
                if len(bars) >= 50 and self.pipeline:
                    last_bars = bars.tail(BACKTEST_PERIOD_BARS)
                    print("[agg] running pipeline demo on latest OHLC closes...")
                    try:
                        result = self.pipeline.train_and_backtest(
                            last_bars,
                            price_col="price",
                            horizon=1,
                            task="classification",
                            initial_train_size=max(50, int(len(last_bars) * 0.5)),
                            step=min(20, max(5, int(len(last_bars) * 0.1))),
                            model_type="classifier",
                            save_model_id=f"live_demo_{int(time.time())}",
                        )
                        print("[agg] pipeline result:", result)
                    except Exception as e:
                        print("[agg] pipeline error:", repr(e))
                await asyncio.wait_for(self._stop_event.wait(), timeout=AGGREGATE_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print("aggregator error:", repr(e))
                await asyncio.sleep(5)

    # ---------- lifecycle ----------
    async def run(self):
        try:
            print("DerivWorker starting with symbols:", self.symbols, " infer_trailing_zeros:", self.infer_trailing_zeros)
            await self.subscribe_all()
            print("Subscriptions created; starting aggregator loop.")
            self._aggregator_task = asyncio.create_task(self.aggregator_loop())
            await self._stop_event.wait()
            print("Stop event received; shutting down worker.")
        finally:
            await self._cleanup()

    async def _cleanup(self):
        print("Cleaning up subscriptions and connections...")
        if self._aggregator_task:
            try:
                self._aggregator_task.cancel()
            except Exception:
                pass
        if self._recv_task:
            try:
                self._recv_task.cancel()
            except Exception:
                pass
        try:
            if self.api and hasattr(self.api, "clear"):
                await self.api.clear()
        except Exception:
            pass
        try:
            if self._raw_conn:
                await self._raw_conn.close()
        except Exception:
            pass
        print("Cleanup done.")

    def stop(self):
        if not self._stop_event.is_set():
            self._stop_event.set()


# ---------- script entrypoint ----------
def _install_signal_handlers(loop, worker: DerivWorker):
    def _sig_handler(sig_num, frame):
        print(f"Signal {sig_num} received -> stopping worker.")
        try:
            loop.call_soon_threadsafe(worker.stop)
        except Exception:
            pass
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig_handler)
        except Exception:
            pass


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                   help="Comma-separated symbols list (e.g. R_100,R_50)")
    p.add_argument("--push-url", type=str, default=DASH_PUSH_URL or "",
                   help="Dashboard push URL (overrides HERO_DASHBOARD_PUSH_URL env)")
    p.add_argument("--ws-url", type=str, default=DERIV_WS_URL, help="Deriv websocket URL")
    p.add_argument("--no-infer-zero", action="store_true", help="Disable trailing-zero inference/padding")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    infer = DEFAULT_INFER_TRAILING_ZEROS and (not args.no_infer_zero)
    if args.push_url:
        DASH_PUSH_URL = args.push_url
    if args.ws_url:
        DERIV_WS_URL = args.ws_url
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = DEFAULT_SYMBOLS

    worker = DerivWorker(symbols=symbols, infer_trailing_zeros=infer)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(loop, worker)

    try:
        print("Starting DerivWorker (async). Press Ctrl+C to stop.")
        loop.run_until_complete(worker.run())
    except KeyboardInterrupt:
        print("KeyboardInterrupt -> stopping")
        worker.stop()
        try:
            loop.run_until_complete(worker._cleanup())
        except Exception:
            pass
    except Exception as e:
        print("Fatal worker error:", repr(e))
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()
        print("Worker exited.")