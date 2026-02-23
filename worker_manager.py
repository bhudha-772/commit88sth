# worker_manager.py -- robust version (auto-detect worker script)
import os, sys, time, signal, subprocess
from pathlib import Path
from typing import Optional, List

PROJECT_DIR = Path(__file__).parent.resolve()
# Do not hardcode one script here; auto-detect below from candidates
WORKER_CANDIDATES = [
    "hero_worker.py",
    "deriv_worker_async.py",
    "hero_worker_async.py",
    "worker.py",
    "deriv_worker.py",
]
PID_FILE = PROJECT_DIR / 'hero_worker.pid'
LOG_FILE = PROJECT_DIR / 'hero_worker.log'

def _resolve_worker_script(explicit: Optional[str] = None) -> Optional[Path]:
    """
    If explicit provided and exists -> use it.
    Otherwise search candidates in PROJECT_DIR and return first existing.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = PROJECT_DIR / explicit
        if p.exists():
            return p.resolve()
        return None
    for name in WORKER_CANDIDATES:
        p = PROJECT_DIR / name
        if p.exists():
            return p.resolve()
    return None

def _read_pid() -> Optional[int]:
    try:
        if not PID_FILE.exists():
            return None
        txt = PID_FILE.read_text().strip()
        if not txt:
            return None
        return int(txt)
    except Exception:
        try:
            PID_FILE.unlink()
        except Exception:
            pass
        return None

def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def is_worker_running() -> bool:
    pid = _read_pid()
    if not pid:
        return False
    if _is_process_alive(pid):
        return True
    try:
        PID_FILE.unlink()
    except Exception:
        pass
    return False


def start_worker(symbols: Optional[List[str]] = None,
                 extra_args: Optional[List[str]] = None,
                 push_url: Optional[str] = None,
                 worker_script: Optional[str] = None) -> dict:
    """
    Start worker as detached process.
    - symbols: list of symbol codes (optional)
    - extra_args: additional CLI args (list or string)
    - push_url: URL that worker should POST ticks to (passes --push-url <url>)
    - worker_script: override script path (relative to project dir or absolute)
    Returns dict with ok/bool and info.
    """
    if is_worker_running():
        return {'ok': False, 'error': 'worker already running', 'pid': _read_pid()}

    resolved = _resolve_worker_script(worker_script)
    if not resolved:
        return {'ok': False, 'error': f'worker script not found (candidates: {WORKER_CANDIDATES})'}

    python = sys.executable or 'python3'
    cmd = [python, str(resolved)]

    # Normalize extra_args: accept str or list and tokenize safely
    extra_args_list = []
    if extra_args:
        if isinstance(extra_args, str):
            try:
                import shlex as _shlex
                extra_args_list = _shlex.split(extra_args)
            except Exception:
                extra_args_list = extra_args.split()
        elif isinstance(extra_args, (list, tuple)):
            extra_args_list = list(extra_args)
        else:
            extra_args_list = [str(extra_args)]

    # If symbols explicitly provided, add them unless extra_args already contains --symbols
    if symbols:
        if isinstance(symbols, (list, tuple)):
            symbols_csv = ",".join([str(s).strip() for s in symbols if s is not None])
        else:
            symbols_csv = str(symbols)
        # don't duplicate if extra_args already contains --symbols
        if not any(a == '--symbols' for a in extra_args_list):
            cmd += ['--symbols', symbols_csv]

    # append extra args (already tokenized)
    if extra_args_list:
        cmd += extra_args_list

    # Append push_url unless already present in extra_args
    if push_url:
        if not any(a == '--push-url' for a in extra_args_list):
            cmd += ['--push-url', str(push_url)]

    # Ensure log available
    try:
        logfile = open(str(LOG_FILE), 'a', buffering=1)
    except Exception as e:
        return {'ok': False, 'error': f'cannot open log file {LOG_FILE}: {e}'}

    try:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_DIR), stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            PID_FILE.write_text(str(proc.pid))
        except Exception:
            pass
        return {'ok': True, 'pid': proc.pid, 'message': 'worker started', 'cmd': cmd, 'log': str(LOG_FILE)}
    except Exception as e:
        try:
            logfile.close()
        except Exception:
            pass
        return {'ok': False, 'error': f'failed to spawn worker: {e}', 'cmd': cmd}
def stop_worker(timeout: float = 5.0) -> dict:
    pid = _read_pid()
    if not pid:
        return {'ok': False, 'error': 'no pidfile (worker not running?)'}
    if not _is_process_alive(pid):
        try:
            PID_FILE.unlink()
        except Exception:
            pass
        return {'ok': True, 'stopped': True, 'pid': pid, 'message': 'not found (cleaned pidfile)'}
    try:
        os.kill(pid, signal.SIGTERM)
    except PermissionError as e:
        return {'ok': False, 'error': f'permission denied SIGTERM pid {pid}: {e}', 'pid': pid}
    except ProcessLookupError:
        try:
            PID_FILE.unlink()
        except Exception:
            pass
        return {'ok': True, 'stopped': True, 'pid': pid}
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if not _is_process_alive(pid):
            try:
                PID_FILE.unlink()
            except Exception:
                pass
            return {'ok': True, 'stopped': True, 'pid': pid, 'message': 'graceful stop'}
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception as e:
        return {'ok': False, 'error': f'failed to SIGKILL pid {pid}: {e}', 'pid': pid}
    try:
        PID_FILE.unlink()
    except Exception:
        pass
    return {'ok': True, 'stopped': True, 'pid': pid, 'message': 'killed'}
