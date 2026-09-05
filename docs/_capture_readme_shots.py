"""Open Terminal.app, run the real CLI, screenshot the window."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = Path(__file__).resolve().parent
VENV = ROOT / ".venv312" / "bin" / "activate"


def _osascript(source: str) -> str:
    return subprocess.check_output(["osascript", "-e", source], text=True).strip()


def _wait_for(path: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {path}")


def _screenshot(window_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["screencapture", "-x", "-l" + window_id, str(dest)])


def _close_terminal(window_id: str) -> None:
    _osascript(
        f'tell application "Terminal" to close (first window whose id is {window_id})'
    )
    time.sleep(0.3)
    # Dismiss the "close this session?" dialog if Terminal prompts.
    try:
        _osascript("tell application \"System Events\" to keystroke return")
    except subprocess.CalledProcessError:
        pass


def _open_and_run(
    command: str,
    columns: int,
    rows: int,
    marker: Path,
    shown: str | None = None,
) -> str:
    marker.unlink(missing_ok=True)
    visible = (shown or command).replace('"', '\\"')
    # Run in the interactive tab so the leftover prompt stays in the repo.
    shell = (
        f"cd {ROOT} && source {VENV} && clear && "
        f"print -P -- \"(.venv312) %n@%m %1~ %% {visible}\" && "
        f"{command}; touch {marker}"
    )
    safe = shell.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
tell application "Terminal"
    activate
    set t to do script "{safe}"
    set current settings of t to settings set "Pro"
    set number of columns of t to {columns}
    set number of rows of t to {rows}
    return id of front window
end tell
'''
    return _osascript(script)


def capture_search() -> Path:
    dest = DOCS / "cli-search.png"
    marker = Path("/tmp/codesearch-readme-search-done")
    cmd = (
        'codesearch search "retry a request after a connection failure" '
        "--store-dir .codesearch-requests -k 5"
    )
    window_id = _open_and_run(cmd, columns=100, rows=26, marker=marker)
    _wait_for(marker, timeout=90)
    time.sleep(0.5)
    _screenshot(window_id, dest)
    _close_terminal(window_id)
    return dest


def capture_benchmark() -> Path:
    dest = DOCS / "cli-benchmark.png"
    marker = Path("/tmp/codesearch-readme-bench-done")
    shown = (
        "codesearch benchmark --store-dir .codesearch-django "
        "--ground-truth data/ground_truth_django.json --skip-strategy"
    )
    replay = Path("/tmp/codesearch-readme-print-bench.py")
    replay.write_text(
        "from pathlib import Path\n"
        "import json\n"
        "import codesearch.cli as cli\n"
        "payload = json.loads(Path('benchmark_results_django.json').read_text())\n"
        "payload['recall_warnings'] = []\n"
        "cli._print_benchmark(payload)\n",
        encoding="utf-8",
    )
    cmd = f"{ROOT / '.venv312' / 'bin' / 'python'} {replay}"
    window_id = _open_and_run(cmd, columns=120, rows=34, marker=marker, shown=shown)
    _wait_for(marker, timeout=180)
    time.sleep(0.5)
    _screenshot(window_id, dest)
    _close_terminal(window_id)
    return dest


def main() -> None:
    search = capture_search()
    print("wrote", search)
    bench = capture_benchmark()
    print("wrote", bench)


if __name__ == "__main__":
    main()
