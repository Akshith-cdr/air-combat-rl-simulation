"""Code provenance saved with every checkpoint and result (reviewer comment M2).

Git is only read here (rev-parse, status). Nothing is staged or committed.
"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Generated outputs live here and do not count as code changes.
OUTPUT_DIRS = ("runs",)


def _git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def git_state():
    """Commit hash and the list of uncommitted code changes (empty = clean)."""
    excludes = [f":(exclude){name}" for name in OUTPUT_DIRS]
    changes = _git("status", "--porcelain", "--", ".", *excludes).splitlines()
    return {"commit": _git("rev-parse", "HEAD").strip(),
            "dirty": bool(changes),
            "uncommitted_changes": changes}


def require_clean_tree(allow_dirty=False):
    """Refuse to produce a reportable result from uncommitted code."""
    state = git_state()
    if state["dirty"] and not allow_dirty:
        listing = "\n  ".join(state["uncommitted_changes"][:20])
        raise SystemExit(
            "Refusing to run: the working tree has uncommitted changes, so the result "
            "could not be tied to a commit.\n  " + listing +
            "\nCommit the code first, or pass --allow-dirty for a non-reportable test run.")
    return state


def package_versions():
    import gymnasium
    import numpy
    import panda3d
    import stable_baselines3
    import torch
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__,
            "gymnasium": gymnasium.__version__, "numpy": numpy.__version__,
            "panda3d": getattr(panda3d, "__version__", "unknown")}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def record(allow_dirty=False):
    """Everything needed to tie an output to the exact code that produced it."""
    from pursuit import config
    return {"git": require_clean_tree(allow_dirty), "versions": package_versions(),
            "task_config": config.as_dict()}
