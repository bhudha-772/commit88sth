#!/usr/bin/env python3
"""
apply_herox_patches.py
Minimal, safe in-place edits for HeroX files:
 - templates/charts.html   : VOTE_THRESHOLD 3 -> 2 and insert auto-trade trigger after voteSignal
 - hero_service.py         : add 'signal' to prediction broadcasts; default stake=1.0 in control_place_trade;
                            insert balance refresh after emitting prediction_result
 - higher_lower_trade_daemon.py : DURATION default 1 -> 5
Backups: file.orig.bak created for each edited file.
"""
import io, os, re, shutil, sys, time

def backup(path):
    bak = path + ".orig.bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"backup: {path} -> {bak}")
    else:
        print(f"backup already exists: {bak}")

def write_if_changed(path, content):
    old = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            old = f.read()
    except FileNotFoundError:
        print(f"ERROR: file not found: {path}")
        return False
    if old == content:
        print(f"no changes for: {path}")
        return False
    # write new
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"patched: {path}")
    return True

def patch_charts_html(path):
    try:
        txt = open(path, "r", encoding="utf-8").read()
    except Exception as e:
        print("charts.html read error:", e); return
    backup(path)

    changed = False
    # 1) VOTE_THRESHOLD 3 -> 2
    if re.search(r"const\s+VOTE_THRESHOLD\s*=\s*3\s*;", txt):
        txt = re.sub(r"const\s+VOTE_THRESHOLD\s*=\s*3\s*;", "const VOTE_THRESHOLD = 2;", txt, count=1)
        changed = True
        print("charts: VOTE_THRESHOLD -> 2")
    else:
        print("charts: VOTE_THRESHOLD not matched or already changed")

    # 2) Insert auto-trade code after "const vs = voteSignal(ind, idx);"
    if "/* --- AUTO-TRADE TRIGGER (vote threshold reached) --- */" in txt:
        print("charts: autotrade snippet already present — skipping insertion")
    else:
        m = re.search(r"(const\s+vs\s*=\s*voteSignal\(ind,\s*idx\);\s*)", txt)
        if m:
            insert_after = m.end(1)
            autotrade = r"""
        // --- AUTO-TRADE TRIGGER (vote threshold reached) ---
        window.__herox_last_signal = window.__herox_last_signal || { sig: null, ts: 0 };
        const last = window.__herox_last_signal;
        const nowTs = Date.now();
        const COOLDOWN_MS = 8000; // 8s cooldown to avoid spamming

        if (vs && (vs.score.long >= VOTE_THRESHOLD || vs.score.short >= VOTE_THRESHOLD)) {
          const signalState = vs.signal === 'long' ? 'higher' : (vs.signal === 'short' ? 'lower' : null);
          if (signalState) {
            if (last.sig !== signalState || (nowTs - (last.ts || 0)) > COOLDOWN_MS) {
              // update last
              window.__herox_last_signal = { sig: signalState, ts: nowTs };

              // Post a place_trade request (include stake=1 for test)
              (async function() {
                try {
                  const payload = {
                    symbol: SYMBOL,
                    signal: signalState,   // explicit 'higher' | 'lower'
                    stake: 1.0             // test stake = 1.0 (server enforces min stake)
                  };
                  const r = await fetch('/control/place_trade', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                  });
                  const j = await r.json().catch(()=>null);
                  rawLog({__meta:'autotrade_sent', payload, resp: j});
                } catch (e) {
                  rawLog({__err:'autotrade_failed', e: String(e)});
                }
              })();
            }
          }
        }
"""
            txt = txt[:insert_after] + autotrade + txt[insert_after:]
            changed = True
            print("charts: inserted autotrade trigger after voteSignal")
        else:
            print("charts: could not find insertion point for autotrade (voteSignal location)")

    if changed:
        write_if_changed(path, txt)

def patch_hero_service(path):
    try:
        txt = open(path, "r", encoding="utf-8").read()
    except Exception as e:
        print("hero_service read error:", e); return
    backup(path)
    orig = txt

    # 1) Insert 'signal' extraction and add signal to two _broadcast_analysis calls inside add_prediction_log
    # We will replace the block starting at the comment about broadcasting with a new block including signal.
    pattern = re.compile(
        r"(\s*# Broadcast the prediction event \(so UI shows it and can toast\)\.\s*\n\s*try:\s*\n\s*_broadcast_analysis\(\{[\s\S]*?\}\)\s*\n\s*_broadcast_analysis\(\{[\s\S]*?\}\)\s*\n\s*except Exception:\s*\n\s*pass)",
        flags=re.M
    )
    if pattern.search(txt):
        replacement = r'''
        # derive signal if provided (normalize common forms)
        signal_val = None
        try:
            if isinstance(payload, dict):
                signal_val = payload.get("signal") or payload.get("direction") or payload.get("trend")
                if signal_val is not None:
                    sv = str(signal_val).strip().lower()
                    if sv in ("1", "higher", "up", "bull"):
                        signal_val = "higher"
                    elif sv in ("-1", "lower", "down", "bear", "downtrend"):
                        signal_val = "lower"
                    else:
                        signal_val = sv
        except Exception:
            signal_val = None

        # Broadcast the prediction event (so UI shows it and can toast)
        try:
            _broadcast_analysis({
                "analysis_event": "prediction_posted",
                "prediction_id": pid,
                "market": sym,
                "symbol": sym,
                "prediction_digit": pred_digit,
                "signal": signal_val,
                "confidence": confidence,
                "stake": chosen_stake if chosen_stake is not None else payload.get("stake"),
                "amount": chosen_stake if chosen_stake is not None else payload.get("amount"),
                "timestamp": ts_now,
                "message": f"Prediction produced for {sym}: {pred_digit}"
            })
            _broadcast_analysis({
                "analysis_event": "prediction_toast",
                "prediction_id": pid,
                "symbol": sym,
                "market": sym,
                "digit": pred_digit,
                "signal": signal_val,
                "stake": chosen_stake if chosen_stake is not None else payload.get("stake"),
                "amount": chosen_stake if chosen_stake is not None else payload.get("amount"),
                "confidence": confidence,
                "status": "produced",
                "message": f"Prediction: {sym} → {pred_digit}"
            })
        except Exception:
            pass
'''
        txt, n = pattern.subn(replacement, txt, count=1)
        if n:
            print("hero_service: added signal extraction & included in broadcasts")
        else:
            print("hero_service: failed to substitute broadcast block (pattern didn't match)")

    else:
        print("hero_service: broadcast pattern not found (maybe already patched)")

    # 2) Default stake behavior in control_place_trade: set stake=1.0 if missing
    # Find the MIN_STAKE block and replace section to set default
    minstake_pat = re.compile(r"(\s*# enforce minimum stake \(server side\)\s*\n\s*MIN_STAKE\s*=\s*0\.35\s*\n\s*if\s+stake\s+is\s+None\s+or\s+stake\s+<\s+MIN_STAKE:\s*\n\s*return\s+jsonify\(\{\"ok\": False, \"error\": \"stake_too_small\", \"min_stake\": MIN_STAKE\}\), 400\s*)",
                              flags=re.M)
    if minstake_pat.search(txt):
        replacement2 = r'''
        # Default stake for quick tests if none provided
        if stake is None:
            stake = 1.0
        # enforce minimum stake (server side)
        MIN_STAKE = 0.35
        if stake is None or stake < MIN_STAKE:
            return jsonify({"ok": False, "error": "stake_too_small", "min_stake": MIN_STAKE}), 400
'''
        txt, n2 = minstake_pat.subn(replacement2, txt, count=1)
        if n2:
            print("hero_service: set default stake to 1.0 when missing (control_place_trade)")
        else:
            print("hero_service: MIN_STAKE block not replaced (pattern mismatch)")
    else:
        # try a more lenient search: locate MIN_STAKE line and insert default stake just above it if not already present
        if re.search(r"MIN_STAKE\s*=\s*0\.35", txt) and "Default stake for quick tests if none provided" not in txt:
            txt = re.sub(r"(\n\s*MIN_STAKE\s*=\s*0\.35)", r"\n        # Default stake for quick tests if none provided\n        if stake is None:\n            stake = 1.0\n\1", txt, count=1)
            print("hero_service: inserted default stake near MIN_STAKE (lenient)")
        else:
            print("hero_service: MIN_STAKE not found or default already present")

    # 3) Insert balance refresh after enqueue_sse(... event='prediction_result')
    # Find the enqueue_sse call for prediction_result and insert balance-refresh block after it, unless already present.
    if '"event": "prediction_result"' in txt or "event=\"prediction_result\"" in txt:
        # We'll search for the pattern of _enqueue_sse({...}, event="prediction_result")
        enqueue_pat = re.compile(r"(_enqueue_sse\(\{\s*[\s\S]*?\}\s*,\s*event\s*=\s*[\"']prediction_result[\"']\)\s*)", flags=re.M)
        m = enqueue_pat.search(txt)
        if m:
            # ensure not already inserted: look for 'balance_update' soon after
            tail = txt[m.end(): m.end()+400]
            if "balance_update" in tail:
                print("hero_service: balance refresh already present after prediction_result enq")
            else:
                balance_block = r'''
        # --- BEST-EFFORT: refresh account balances for demo/real and broadcast balance_update ---
        try:
            for m in ("demo", "real"):
                mgr = _accounts.get(m)
                if mgr and mgr.authorized:
                    try:
                        bal_res = mgr.get_balance(timeout=6.0)
                        if isinstance(bal_res, dict) and bal_res.get("ok"):
                            try:
                                _broadcast_analysis({
                                    "analysis_event": "balance_update",
                                    "mode": m,
                                    "balance": bal_res.get("balance"),
                                    "raw": bal_res.get("raw")
                                })
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass
'''
                ins_point = m.end(1)
                txt = txt[:ins_point] + balance_block + txt[ins_point:]
                print("hero_service: inserted balance refresh after prediction_result enqueue")
        else:
            print("hero_service: could not find _enqueue_sse(..., event='prediction_result') pattern")
    else:
        print("hero_service: prediction_result enqueue pattern not found or already changed")

    if txt != orig:
        write_if_changed(path, txt)
    else:
        print("hero_service: no textual changes applied")

def patch_hl_daemon(path):
    try:
        txt = open(path, "r", encoding="utf-8").read()
    except Exception as e:
        print("hl daemon read error:", e); return
    backup(path)
    # replace HL_DURATION default "1" -> "5"
    if re.search(r'HL_DURATION"\s*,\s*"1"\)', txt) or re.search(r'HL_DURATION",\s*"1"\)', txt) or re.search(r'HL_DURATION",\s*"1"\)', txt):
        txt2 = txt
        txt2 = re.sub(r'(DURATION\s*=\s*int\(os\.environ\.get\("HL_DURATION"\s*,\s*")1("\)\))',
                      r'\1'.replace('1','5'), txt2)  # fallback: complex; we'll do simpler
    # simpler approach: find the DURATION assignment and replace the default string "1" with "5"
    newtxt = re.sub(r'DURATION\s*=\s*int\s*\(\s*os\.environ\.get\s*\(\s*"HL_DURATION"\s*,\s*"[0-9]+"\s*\)\s*\)',
                    'DURATION = int(os.environ.get("HL_DURATION", "5"))', txt, count=1)
    if newtxt != txt:
        write_if_changed(path, newtxt)
        print("higher_lower_trade_daemon: set default DURATION to 5 ticks")
    else:
        # maybe already 5 or pattern different
        if re.search(r'DURATION\s*=\s*int\s*\(\s*os\.environ\.get\s*\(\s*"HL_DURATION"\s*,\s*"\s*5\s*"\s*\)\s*\)', txt):
            print("higher_lower_trade_daemon: DURATION already 5 — no change")
        else:
            print("higher_lower_trade_daemon: DURATION pattern not matched — manual check advised")

def main():
    # file paths (adjust if your tree differs)
    chart_paths = [
        "templates/charts.html", "charts.html", "templates/charts.htm"
    ]
    hero_paths = ["hero_service.py", "src/hero_service.py"]
    hl_paths = ["higher_lower_trade_daemon.py", "higher_lower_trade_daemon/higher_lower_trade_daemon.py"]

    # find first existing chart file
    chart_file = next((p for p in chart_paths if os.path.exists(p)), None)
    hero_file = next((p for p in hero_paths if os.path.exists(p)), None)
    hl_file = next((p for p in hl_paths if os.path.exists(p)), None)

    if not chart_file:
        print("WARNING: charts.html not found in expected locations. Skipping chart patch.")
    else:
        patch_charts_html(chart_file)

    if not hero_file:
        print("WARNING: hero_service.py not found in expected locations. Skipping hero_service patch.")
    else:
        patch_hero_service(hero_file)

    if not hl_file:
        print("WARNING: higher_lower_trade_daemon.py not found in expected locations. Skipping HL patch.")
    else:
        patch_hl_daemon(hl_file)

    print("\\nPatching finished. Inspect backups (*.orig.bak) and test.\\n")

if __name__ == '__main__':
    main()
