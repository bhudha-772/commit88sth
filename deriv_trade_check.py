#!/usr/bin/env python3
# deriv_trade_check.py
# Quick check: authorize -> balance -> proposal -> buy
# Usage: set DERIV_TOKEN env var (or pass --token), then run.
# Requires: pip install websocket-client

import os
import json
import time
import logging
import argparse
import threading
from websocket import WebSocketApp

# --- Configuration (env overrides) ---
DEFAULT_APP_ID = int(os.environ.get("DERIV_APP_ID", os.environ.get("APP_ID", "71710")))
DEFAULT_WS = os.environ.get("DERIV_WS_URL", f"wss://ws.binaryws.com/websockets/v3?app_id={DEFAULT_APP_ID}")
DEFAULT_SYMBOL = os.environ.get("DERIV_SYMBOL", "R_10")
DEFAULT_AMOUNT = float(os.environ.get("DERIV_AMOUNT", "1"))
DEFAULT_DURATION = int(os.environ.get("DERIV_DURATION", "60"))
DEFAULT_DURATION_UNIT = os.environ.get("DERIV_DURATION_UNIT", "s")  # "s" or "m"
DEFAULT_CONTRACT_TYPE = os.environ.get("DERIV_CONTRACT_TYPE", "CALL")  # CALL or PUT etc
DEFAULT_CURRENCY = os.environ.get("DERIV_CURRENCY", "USD")
LOG_FILE = os.environ.get("DERIV_LOG_FILE", "deriv_trade_check.log")

# --- Logging ---
logger = logging.getLogger("deriv_check")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
logger.addHandler(fh)

# --- Sync primitives to coordinate events between callbacks and main thread ---
authorized_event = threading.Event()
proposal_event = threading.Event()
buy_event = threading.Event()

result_container = {
    "balance": None,
    "proposal": None,
    "buy": None,
    "error": None,
}

# Container for proposal id/price to buy
proposal_info = {"id": None, "price": None}

def safe_json(obj):
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)

def build_proposal_payload(symbol, amount, duration, duration_unit, contract_type, currency):
    return {
        "proposal": 1,
        "amount": amount,
        "basis": "stake",      # we use stake (you can change to 'payout' if needed)
        "contract_type": contract_type,
        "currency": currency,
        "duration": duration,
        "duration_unit": duration_unit,
        "symbol": symbol,
    }

def _extract_price_from_proposal(prop: dict):
    """Try common fields where the server returns a numeric price for buy."""
    if not prop:
        return None
    for k in ("ask_price", "price", "display_value", "display_value_raw", "payout"):
        if k in prop and prop.get(k) is not None:
            try:
                return float(prop.get(k))
            except Exception:
                # sometimes display_value is a string like "1.00" — float() handles that
                try:
                    # if it's nested, try converting string
                    return float(str(prop.get(k)))
                except Exception:
                    continue
    return None

def on_open(ws):
    logger.info("WS open -> sending authorize")
    token = ws.app_token
    if not token:
        logger.error("No token provided to WebSocket app (DERIV_TOKEN missing).")
        ws.close()
        return
    auth_msg = {"authorize": token}
    try:
        ws.send(json.dumps(auth_msg))
        logger.debug("Sent authorize: %s", safe_json(auth_msg))
    except Exception as e:
        logger.exception("Failed to send authorize: %s", e)
        result_container["error"] = f"send_authorize_failed: {e}"
        ws.close()

def on_message(ws, message):
    logger.debug("Raw message recv: %s", message)
    try:
        data = json.loads(message)
    except Exception as e:
        logger.exception("Failed to parse JSON message: %s", e)
        return

    # Top-level error handling
    if "error" in data:
        logger.error("API returned error: %s", safe_json(data.get("error")))
        result_container["error"] = data.get("error")
        # set events so main thread can bail out
        authorized_event.set()
        proposal_event.set()
        buy_event.set()
        return

    # AUTHORIZE response
    if "authorize" in data:
        logger.info("Authorized OK -> requesting balance and proposal")
        authorized_event.set()
        # request balance
        try:
            ws.send(json.dumps({"balance": 1}))
            logger.debug("Sent balance request")
        except Exception:
            logger.exception("Failed to send balance request")
        # also send proposal immediately (we'll attempt to buy on response)
        try:
            prop = build_proposal_payload(
                ws.app_symbol, ws.app_amount, ws.app_duration, ws.app_duration_unit, ws.app_contract_type, ws.app_currency
            )
            ws.send(json.dumps(prop))
            logger.info("Sent proposal request: %s", safe_json(prop))
        except Exception:
            logger.exception("Failed to send proposal request")

        return

    # BALANCE response
    if "balance" in data:
        logger.info("Balance response: %s", safe_json(data.get("balance")))
        result_container["balance"] = data.get("balance")
        return

    # PROPOSAL response (server provides pricing)
    if "proposal" in data:
        prop = data.get("proposal")
        logger.info("Received proposal: %s", safe_json(prop))
        proposal_info["id"] = prop.get("id") or prop.get("proposal_id") or prop.get("proposal")
        # price fields vary: ask_price, price, display_value
        extracted_price = _extract_price_from_proposal(prop)
        proposal_info["price"] = extracted_price
        result_container["proposal"] = prop
        proposal_event.set()

        # Attempt to buy the proposal id (standard flow)
        if proposal_info["id"]:
            # --- FIX: include the required "price" property when sending buy ---
            buy_payload = {"buy": proposal_info["id"]}
            if proposal_info["price"] is not None:
                try:
                    buy_payload["price"] = float(proposal_info["price"])
                except Exception:
                    # fallback to string (server usually accepts numeric)
                    buy_payload["price"] = proposal_info["price"]
            else:
                # if we don't have a price, try to use display_value or fallback to amount
                logger.warning("Proposal did not include a clear price; attempting to extract/display fallback.")
            try:
                logger.info("Attempting buy for proposal id=%s price=%s", proposal_info["id"], buy_payload.get("price"))
                ws.send(json.dumps(buy_payload))
                logger.debug("Sent buy: %s", safe_json(buy_payload))
            except Exception:
                logger.exception("Failed to send buy payload")
        else:
            logger.error("No proposal id provided by server; cannot buy.")
        return

    # BUY response
    if "buy" in data:
        logger.info("Buy response: %s", safe_json(data.get("buy")))
        result_container["buy"] = data.get("buy")
        buy_event.set()
        return

    # Some servers reply with 'proposal_open_contract' or other keys — log them
    logger.debug("Unhandled message keys: %s", list(data.keys()))
    return

def on_error(ws, error):
    logger.exception("WebSocket error: %s", error)
    result_container["error"] = str(error)
    # wake main thread
    authorized_event.set()
    proposal_event.set()
    buy_event.set()

def on_close(ws, close_status_code, close_msg):
    logger.info("WS closed. code=%s msg=%s", close_status_code, close_msg)

def run_trade_check(ws_url, token, symbol, amount, duration, duration_unit, contract_type, currency, timeout_total=30):
    logger.info("Starting trade check: ws=%s symbol=%s amount=%s duration=%s%s type=%s",
                ws_url, symbol, amount, duration, duration_unit, contract_type)

    # Prepare WebSocketApp and attach config
    ws_app = WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    # Attach custom attributes for callbacks to use
    ws_app.app_token = token
    ws_app.app_symbol = symbol
    ws_app.app_amount = amount
    ws_app.app_duration = duration
    ws_app.app_duration_unit = duration_unit
    ws_app.app_contract_type = contract_type
    ws_app.app_currency = currency

    # Run WS in background thread so main thread can implement timeouts
    ws_thread = threading.Thread(target=lambda: ws_app.run_forever(ping_interval=30, ping_timeout=10), daemon=True)
    ws_thread.start()

    start = time.time()
    # Wait for auth (but timeout if nothing)
    if not authorized_event.wait(timeout=10):
        logger.error("Authorization did not complete within 10s. Aborting.")
        result_container["error"] = "authorize_timeout"
        try:
            ws_app.close()
        except Exception:
            pass
        return result_container

    # Wait for proposal and buy result (or error) within the remaining timeout
    remaining = max(0, timeout_total - (time.time() - start))
    logger.debug("Waiting up to %s seconds for buy result", remaining)
    buy_done = buy_event.wait(timeout=remaining)
    if not buy_done:
        logger.error("Buy did not complete within timeout (%s s). Proposal present=%s", timeout_total, bool(result_container.get("proposal")))
        result_container["error"] = result_container.get("error") or "buy_timeout"
    else:
        logger.info("Buy completed; see result_container['buy'] for details")

    # close websocket cleanly
    try:
        ws_app.close()
    except Exception:
        pass
    # join thread briefly
    ws_thread.join(timeout=2.0)
    return result_container

def main():
    parser = argparse.ArgumentParser(description="Deriv quick trade check: authorize->balance->proposal->buy")
    parser.add_argument("--token", "-t", default=os.environ.get("DERIV_TOKEN"), help="Deriv API token (or set DERIV_TOKEN env var)")
    parser.add_argument("--ws", default=DEFAULT_WS, help="Deriv websocket URL")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--amount", type=float, default=DEFAULT_AMOUNT)
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    parser.add_argument("--duration-unit", default=DEFAULT_DURATION_UNIT)
    parser.add_argument("--contract-type", default=DEFAULT_CONTRACT_TYPE)
    parser.add_argument("--currency", default=DEFAULT_CURRENCY)
    parser.add_argument("--timeout", type=int, default=30, help="Total timeout seconds for the flow")
    args = parser.parse_args()

    if not args.token:
        logger.error("No DERIV token provided. Set DERIV_TOKEN env var or use --token.")
        return

    # Clear any previous state events
    authorized_event.clear()
    proposal_event.clear()
    buy_event.clear()

    res = run_trade_check(
        ws_url=args.ws,
        token=args.token,
        symbol=args.symbol,
        amount=args.amount,
        duration=args.duration,
        duration_unit=args.duration_unit,
        contract_type=args.contract_type,
        currency=args.currency,
        timeout_total=args.timeout,
    )

    logger.info("=== SUMMARY ===")
    logger.info("Balance: %s", safe_json(res.get("balance")))
    logger.info("Proposal: %s", safe_json(res.get("proposal")))
    logger.info("Buy: %s", safe_json(res.get("buy")))
    logger.info("Error (if any): %s", safe_json(res.get("error")))

if __name__ == "__main__":
    main()