#!/usr/bin/env python3
# telegram_bot.py - HeroX Telegram controller + conversation integration.
from __future__ import annotations
import argparse
import logging
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from typing import Optional
import json

import requests
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# local modules
from memory_local import MemoryLocal
from conversation import ConversationManager
from ingest import ingest_url  # direct import so handlers can call

LOG = logging.getLogger("hero_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# DEV TOKEN EMBED — kept as requested
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN") or "8261836397:AAG4OrFKdLs2KmY4LIRUvyCzMq67apGV3Y8"

DEFAULT_PORT = 5000
GLOBAL_STATE = {"dashboard_url": None, "dashboard_manager": None}

# create memory and conversation engine (shared)
MEM = MemoryLocal()          # uses hero_memory.db by default (see memory_local.py)
conv = ConversationManager(memory=MEM)


def get_local_ip(timeout: float = 0.3) -> Optional[str]:
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(timeout)
        s.connect(("1.1.1.1", 80))
        ip = s.getsockname()[0]
        if not ip or ip.startswith("127.") or ip.startswith("169.254.") or ip == "0.0.0.0":
            return None
        return ip
    except Exception:
        return None
    finally:
        try:
            if s:
                s.close()
        except Exception:
            pass


class DashboardManager:
    def __init__(self, venv_python: Optional[str] = None):
        self._server_obj = None
        self._proc = None
        self.port = DEFAULT_PORT
        self.host = None
        self.venv_python = venv_python or sys.executable

    def start(self, host: str, port: Optional[int] = None, open_browser: bool = False) -> str:
        if not host or host in ("0.0.0.0", "localhost") or host.startswith("127."):
            raise RuntimeError(f"Refusing to start dashboard bound to disallowed host: {host!r}")

        if port is not None:
            self.port = int(port)
        bind_host = host
        self.host = bind_host

        # try in-process first
        try:
            import hero_service  # type: ignore
            if hasattr(hero_service, "DashboardServer"):
                LOG.info("Starting hero_service.DashboardServer in-process (bind_host=%s)", bind_host)
                ds = hero_service.DashboardServer(host=bind_host, port=self.port, open_browser=open_browser)
                try:
                    ds.start()
                except Exception:
                    if hasattr(ds, "run"):
                        threading.Thread(target=ds.run, daemon=True).start()
                    elif hasattr(ds, "start_background"):
                        ds.start_background()
                    elif hasattr(ds, "restart"):
                        threading.Thread(target=ds.restart, daemon=True).start()
                self._server_obj = ds
                url = f"http://{bind_host}:{self.port}/"
                GLOBAL_STATE["dashboard_url"] = url
                LOG.info("Dashboard started in-process: %s", url)
                return url
        except Exception as e:
            LOG.info("Could not use hero_service in-process: %s", e)

        # fallback to subprocess
        script_path = os.path.join(os.getcwd(), "hero_service.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"hero_service.py not found at {script_path}")

        cmd = [self.venv_python, script_path, "--host", bind_host, "--port", str(self.port)]
        LOG.info("Starting hero_service.py via subprocess: %s", " ".join(shlex.quote(x) for x in cmd))

        try:
            os.makedirs(os.path.expanduser("~/.hero_logs"), exist_ok=True)
            stdout = open(os.path.expanduser("~/.hero_logs/dashboard_subproc.out"), "a")
            stderr = open(os.path.expanduser("~/.hero_logs/dashboard_subproc.err"), "a")
        except Exception:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL

        self._proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, start_new_session=True)
        url = f"http://{bind_host}:{self.port}/"
        GLOBAL_STATE["dashboard_url"] = url
        LOG.info("Dashboard started as subprocess (pid=%s): %s", getattr(self._proc, "pid", None), url)
        return url

    def stop(self) -> bool:
        ok = False
        if self._server_obj is not None:
            for meth in ("shutdown", "stop", "restart", "close"):
                fn = getattr(self._server_obj, meth, None)
                if callable(fn):
                    try:
                        fn()
                        ok = True
                    except Exception:
                        LOG.exception("server_obj.%s() raised", meth)
            self._server_obj = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except Exception:
                    self._proc.kill()
                ok = True
            except Exception:
                LOG.exception("Failed to terminate subprocess")
            self._proc = None
        if ok:
            GLOBAL_STATE["dashboard_url"] = None
        return ok

    def is_running(self) -> bool:
        if self._server_obj is not None:
            try:
                return bool(getattr(self._server_obj, "_srv", None) is not None)
            except Exception:
                return True
        if self._proc is not None:
            return self._proc.poll() is None
        return False


# ---------- Telegram command handlers ----------

def cmd_start(update: Update, context: CallbackContext):
    update.message.reply_text("Hello — I'm HeroX. I can chat and manage a dashboard. Try saying 'hi' or use /opendashboard.")

def cmd_status(update: Update, context: CallbackContext):
    dm: DashboardManager = GLOBAL_STATE.get("dashboard_manager")
    running = dm.is_running() if dm else False
    url = GLOBAL_STATE.get("dashboard_url") or "(no dashboard URL known)"
    update.message.reply_text(f"Dashboard running: {running}\nURL: {url}")

def cmd_opendashboard(update: Update, context: CallbackContext):
    dm: DashboardManager = GLOBAL_STATE.get("dashboard_manager")
    if dm is None:
        dm = DashboardManager(venv_python=sys.executable)
        dm.port = DEFAULT_PORT
        GLOBAL_STATE["dashboard_manager"] = dm

    ip = get_local_ip()
    if not ip:
        update.message.reply_text("❌ Could not detect local Wi-Fi IP. Connect to Wi-Fi or use /opendashboard --host <ip> (if implemented).")
        return

    update.message.reply_text(f"Detected local IP: {ip}\nAttempting to start dashboard bound to {ip}:{dm.port} ...")
    try:
        url = dm.start(host=ip, port=dm.port, open_browser=False)
        update.message.reply_text(f"✅ Dashboard started — accessible at:\n{url}")
    except Exception as e:
        LOG.exception("Failed to start dashboard")
        update.message.reply_text(f"Failed to start dashboard: {e}")

def cmd_closedashboard(update: Update, context: CallbackContext):
    dm: DashboardManager = GLOBAL_STATE.get("dashboard_manager")
    if dm is None:
        update.message.reply_text("No dashboard manager configured.")
        return
    ok = dm.stop()
    update.message.reply_text("Dashboard stop requested." if ok else "No dashboard running.")

def cmd_dashboardurl(update: Update, context: CallbackContext):
    url = GLOBAL_STATE.get("dashboard_url")
    if url:
        update.message.reply_text(f"Dashboard URL: {url}")
    else:
        update.message.reply_text("Dashboard not started yet.")

def cmd_stopbot(update: Update, context: CallbackContext):
    update.message.reply_text("Shutting down bot and stopping dashboard...")
    dm: DashboardManager = GLOBAL_STATE.get("dashboard_manager")
    if dm:
        dm.stop()
    def _stop():
        time.sleep(0.25)
        os._exit(0)
    threading.Thread(target=_stop, daemon=True).start()

# conversation message handler
def handle_text(update: Update, context: CallbackContext):
    try:
        if not update.message or not update.message.text:
            return
        uid = update.effective_user.id if update.effective_user else 0
        text = update.message.text.strip()
        LOG.info("Message from %s: %s", uid, text)
        reply = conv.handle_message(uid, text)
        update.message.reply_text(reply)
    except Exception:
        LOG.exception("Error in handle_text")
        try:
            update.message.reply_text("Sorry — something went wrong while processing that.")
        except Exception:
            pass

# ingest URL directly into memory (uses background thread + progress callback)
def cmd_ingesturl(update: Update, context: CallbackContext):
    if not context.args:
        update.message.reply_text("Usage: /ingesturl <url>")
        return
    url = context.args[0].strip()
    msg = update.message.reply_text(f"Ingesting {url} ...")

    def progress(p):
        try:
            msg.edit_text(f"Ingesting {url} ...\n{p}")
        except Exception:
            try:
                update.message.reply_text(f"[ingest] {p}")
            except Exception:
                pass

    def _run_ingest():
        try:
            res = ingest_url(url, memory=MEM, progress=progress, ignore_robots=False)
            if res.get("ok"):
                added = res.get("added", 0)
                total = res.get("total_candidates", 0)
                title = res.get("title") or ""
                try:
                    msg.edit_text(f"Ingest finished: {added} new chunks stored (candidates: {total}). Title: {title}")
                except Exception:
                    try:
                        update.message.reply_text(f"Ingest finished: {added} new chunks stored (candidates: {total}). Title: {title}")
                    except Exception:
                        pass
            else:
                err = res.get("error", "unknown")
                try:
                    msg.edit_text(f"Ingest failed: {err}")
                except Exception:
                    try:
                        update.message.reply_text(f"Ingest failed: {err}")
                    except Exception:
                        pass
        except Exception as e:
            LOG.exception("ingest failed")
            try:
                msg.edit_text(f"Ingest exception: {e}")
            except Exception:
                try:
                    update.message.reply_text(f"Ingest exception: {e}")
                except Exception:
                    pass

    threading.Thread(target=_run_ingest, daemon=True).start()

def cmd_recall(update: Update, context: CallbackContext):
    q = " ".join(context.args) if context.args else ""
    if not q:
        update.message.reply_text("Usage: /recall <query>")
        return
    rows = MEM.search(q, limit=6)
    if not rows:
        update.message.reply_text(f"No results for: {q}")
        return
    for doc_id, src, ts, snippet in rows:
        ts_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
        msg = f"#{doc_id} ({src}) [{ts_str}]\n{snippet[:1600]}"
        try:
            update.message.reply_text(msg)
        except Exception:
            pass

def cmd_memstats(update: Update, context: CallbackContext):
    try:
        count = MEM.count()
        update.message.reply_text(f"Memory items: {count}")
    except Exception:
        LOG.exception("memstats failed")
        update.message.reply_text("Failed to fetch memory stats.")

def cmd_teach(update: Update, context: CallbackContext):
    if not context.args:
        update.message.reply_text("Usage: /teach <topic>\nExample: /teach Forex")
        return
    topic = " ".join(context.args).strip()
    update.message.reply_text(f"Preparing to learn about: {topic}")

    dm = GLOBAL_STATE.get("dashboard_manager")
    if dm is None:
        dm = DashboardManager(venv_python=sys.executable)
        dm.port = DEFAULT_PORT
        GLOBAL_STATE["dashboard_manager"] = dm

    if not dm.is_running():
        ip = get_local_ip()
        if ip:
            try:
                update.message.reply_text(f"Starting dashboard bound to {ip}:{dm.port} ...")
                url = dm.start(host=ip, port=dm.port, open_browser=False)
                update.message.reply_text(f"Dashboard started — open here: {url}")
            except Exception as e:
                LOG.exception("dashboard start failed")
                update.message.reply_text(f"Failed to start dashboard: {e}")
        else:
            update.message.reply_text("Dashboard not running and I couldn't detect local IP — continuing without dashboard.")

    wiki_url = "https://en.wikipedia.org/wiki/" + topic.replace(" ", "_")
    update.message.reply_text(f"I will attempt to ingest: {wiki_url}\n(If Wikipedia doesn't have the topic, try /ingesturl with a specific URL.)")

    chat_id = update.effective_chat.id
    bot = context.bot
    status_msg = update.message.reply_text(f"Teaching: ingesting {wiki_url} ...")

    def progress_cb(msg: str):
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text=f"[teach] {msg}")
        except Exception:
            try:
                bot.send_message(chat_id=chat_id, text=f"[teach] {msg}")
            except Exception:
                pass

    def _run_teach():
        try:
            res = ingest_url(wiki_url, memory=MEM, progress=progress_cb, max_chunks=120, ignore_robots=False)
            if res.get("ok"):
                bot.send_message(chat_id=chat_id, text=f"Teach finished. Added: {res.get('added',0)} chunks. Title: {res.get('title')}")
            else:
                bot.send_message(chat_id=chat_id, text=f"Teach failed: {res.get('error')}")
        except Exception as e:
            LOG.exception("teach ingest failed")
            try:
                bot.send_message(chat_id=chat_id, text=f"Teach exception: {e}")
            except Exception:
                pass

    threading.Thread(target=_run_teach, daemon=True).start()

# --- Deriv control commands
def cmd_startderiv(update: Update, context: CallbackContext):
    dm: DashboardManager = GLOBAL_STATE.get("dashboard_manager")
    if dm is None:
        dm = DashboardManager(venv_python=sys.executable)
        dm.port = DEFAULT_PORT
        GLOBAL_STATE["dashboard_manager"] = dm

    if not dm.is_running():
        ip = get_local_ip()
        if not ip:
            update.message.reply_text("Dashboard not running and I couldn't detect IP. Start dashboard with /opendashboard first.")
            return
        update.message.reply_text(f"Starting dashboard at {ip}:{dm.port} ...")
        try:
            url = dm.start(host=ip, port=dm.port, open_browser=False)
            update.message.reply_text(f"Dashboard started: {url}")
        except Exception as e:
            LOG.exception("dashboard start failed")
            update.message.reply_text(f"Failed to start dashboard: {e}")
            return

    dashboard_url = GLOBAL_STATE.get("dashboard_url")
    if not dashboard_url:
        update.message.reply_text("Dashboard URL unknown; cannot start deriv worker.")
        return

    try:
        post_url = dashboard_url.rstrip("/") + "/control/start_deriv"
        r = requests.post(post_url, timeout=6)
        try:
            j = r.json()
        except Exception:
            j = {"status_code": r.status_code, "text": r.text[:400]}
        update.message.reply_text(f"Start deriv response: {j}")
    except Exception as e:
        LOG.exception("startderiv failed")
        update.message.reply_text(f"Failed to trigger start: {e}")

def cmd_stopderiv(update: Update, context: CallbackContext):
    dashboard_url = GLOBAL_STATE.get("dashboard_url")
    if not dashboard_url:
        update.message.reply_text("Dashboard URL unknown; nothing to stop.")
        return
    try:
        post_url = dashboard_url.rstrip("/") + "/control/stop_deriv"
        r = requests.post(post_url, timeout=6)
        try:
            j = r.json()
        except Exception:
            j = {"status_code": r.status_code, "text": r.text[:400]}
        update.message.reply_text(f"Stop deriv response: {j}")
    except Exception as e:
        LOG.exception("stopderiv failed")
        update.message.reply_text(f"Failed to trigger stop: {e}")

def cmd_derivstatus(update: Update, context: CallbackContext):
    dashboard_url = GLOBAL_STATE.get("dashboard_url")
    msg = "Dashboard URL unknown"
    if dashboard_url:
        try:
            r = requests.get(dashboard_url.rstrip("/") + "/recent_ticks", timeout=4)
            try:
                j = r.json()
                ticks = j.get("ticks", [])
                msg = f"Recent ticks (count={len(ticks)}). Latest: {ticks[0] if ticks else 'none'}"
            except Exception:
                msg = f"recent_ticks fetch: status {r.status_code}"
        except Exception as e:
            msg = f"Error contacting dashboard: {e}"
    update.message.reply_text(msg)

# ---------- main/boot ----------

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--token", help="Telegram bot token (or set TELEGRAM_TOKEN env var)")
    p.add_argument("--autostart", action="store_true", help="Start dashboard automatically on bot start")
    p.add_argument("--host", help="Optional host to bind dashboard to (overrides auto-detect)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Dashboard port")
    return p

def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    args = build_argparser().parse_args(argv)

    token = args.token or TELEGRAM_TOKEN or os.environ.get("TELEGRAM_TOKEN")
    if not token:
        LOG.error("No TELEGRAM token provided")
        sys.exit(1)

    dm = DashboardManager(venv_python=sys.executable)
    dm.port = args.port
    GLOBAL_STATE["dashboard_manager"] = dm

    updater = Updater(token=token, use_context=True)
    try:
        me = updater.bot.get_me()
        LOG.info("Bot running as: @%s (id=%s)", me.username, me.id)
    except Exception:
        LOG.exception("Failed to fetch bot identity")

    dp = updater.dispatcher

    # command handlers
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("opendashboard", cmd_opendashboard))
    dp.add_handler(CommandHandler("closedashboard", cmd_closedashboard))
    dp.add_handler(CommandHandler("dashboardurl", cmd_dashboardurl))
    dp.add_handler(CommandHandler("stopbot", cmd_stopbot))

    dp.add_handler(CommandHandler("ingesturl", cmd_ingesturl))
    dp.add_handler(CommandHandler("recall", cmd_recall))
    dp.add_handler(CommandHandler("memorystats", cmd_memstats))
    dp.add_handler(CommandHandler("teach", cmd_teach))

    dp.add_handler(CommandHandler("startderiv", cmd_startderiv))
    dp.add_handler(CommandHandler("stopderiv", cmd_stopderiv))
    dp.add_handler(CommandHandler("derivstatus", cmd_derivstatus))

    # plain-text messages -> conversation
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))

    updater.start_polling()
    LOG.info("HeroX bot started and polling.")
    updater.idle()

if __name__ == "__main__":
    main()
