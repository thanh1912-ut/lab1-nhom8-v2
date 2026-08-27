"""Git auto-backup: commit & push important artifacts to GitHub from Kaggle.

Reads GITHUB_TOKEN from env (Kaggle Secrets attach as env var).
Pushes: results.json, run logs, metric PNGs (cm_best/cm_test), and best.pt of
the best run per (model, optimizer). Large files (ckpt.pt, last.pt) are
gitignored so the repo stays small.
"""
import os
import subprocess

try:
    from . import logger as L
except ImportError:
    import logger as L

MARKER = ".gitbackup_init_done"


def _run(cmd, cwd, check=True, capture=True):
    r = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                       capture_output=capture, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{r.stdout}\n{r.stderr}")
    return r


def _repo_root(cfg):
    # outputs dir sits inside the repo root (parent of outputs dir name)
    return os.path.abspath(os.path.join(cfg["output_dir"], os.pardir))


def _setup_remote(cfg, root):
    gb = cfg.get("git_backup", {})
    token = os.environ.get("GITHUB_TOKEN", "")
    repo_url = gb.get("repo_url", "")
    branch = gb.get("branch", "main")
    if not token:
        print(L.colorize(
            "git backup: GITHUB_TOKEN not set - push needs a token even for "
            "public repos (clone/pull does not). Skipping backup.", L.YELLOW))
        return None
    auth_url = (repo_url.replace("https://", f"https://{token}@")
                if repo_url else None)
    # ensure git identity
    _run(["git", "config", "user.email", "lab1-bot@users.noreply.github.com"],
         root, check=False)
    _run(["git", "config", "user.name", "lab1-bot"], root, check=False)
    # ensure origin
    origin = _run(["git", "remote", "get-url", "origin"], root,
                  check=False, capture=True)
    if origin.returncode != 0:
        url = auth_url or repo_url or ""
        if url:
            _run(["git", "remote", "add", "origin", url], root, check=False)
    elif auth_url:
        _run(["git", "remote", "set-url", "origin", auth_url], root,
             check=False)
    return branch


def _force_add_artifacts(cfg, root):
    """Add results*.json + per-run artifacts (logs, cm PNGs, winner best.pt)."""
    out = cfg["output_dir"]
    paths = []
    runs_dir = os.path.join(out, "runs")
    if os.path.isdir(runs_dir):
        for rid in sorted(os.listdir(runs_dir)):
            rd = os.path.join(runs_dir, rid)
            for f in ("train.log", "cm_best.png", "cm_test.png"):
                p = os.path.join(rd, f)
                if os.path.exists(p):
                    paths.append(p)
    # winner best.pt per (model, optimizer): lowest val loss among seeds
    import json
    try:
        from benchmark import all_results_paths
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from benchmark import all_results_paths
    winners = {}
    for rp in all_results_paths(cfg):
        with open(rp, encoding="utf-8") as f:
            results = json.load(f)
        for rid, h in results.items():
            if not h.get("done"):
                continue
            # v2 runs store best_val_metric (the configured selection metric,
            # e.g. f1 - higher is better). Fall back to best_val_loss
            # (lower is better) for v1-style runs.
            k = (h["model"], h["optimizer"])
            sel = h.get("best_val_metric")
            if sel is not None:
                score = sel
                better = (k not in winners) or (score > winners[k][1])
            elif h.get("best_val_loss") is not None:
                score = h["best_val_loss"]
                better = (k not in winners) or (score < winners[k][1])
            else:
                continue
            if better:
                winners[k] = (rid, score)
    paths.extend(all_results_paths(cfg))
    for (rid, _) in winners.values():
        p = os.path.join(runs_dir, rid, "best.pt")
        if os.path.exists(p):
            paths.append(p)
    add = [p for p in paths if os.path.exists(p)]
    if add:
        _run(["git", "add", "-f", "--"] + add, root)


def _acquire_lock(cfg, timeout=300, stale_sec=900):
    """Serialize git operations between concurrent training processes.
    Returns True if lock acquired (caller must release), False otherwise."""
    import time
    lock = os.path.join(cfg["output_dir"], ".gitbackup.lock")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            # stale lock (holder crashed) -> break it after stale_sec
            try:
                if time.time() - os.path.getmtime(lock) > stale_sec:
                    os.remove(lock)
                    continue
            except OSError:
                pass
            time.sleep(2)
    return False


def _release_lock(cfg):
    lock = os.path.join(cfg["output_dir"], ".gitbackup.lock")
    try:
        os.remove(lock)
    except OSError:
        pass


def backup(cfg, results_path=None, msg=None):
    """Commit & push artifacts. Safe to call frequently; no-op if nothing
    changed. Lock-protected so parallel processes don't corrupt git state."""
    gb = cfg.get("git_backup", {})
    if not gb.get("enabled", False):
        return False
    root = _repo_root(cfg)
    if not os.path.isdir(os.path.join(root, ".git")):
        print(L.colorize("git backup skipped: not a git repo", L.YELLOW))
        return False
    if not os.environ.get("GITHUB_TOKEN"):
        print(L.colorize(
            "git backup skipped: GITHUB_TOKEN not set. "
            "Push (write) needs a token even for public repos - "
            "add it in Kaggle Secrets to enable auto-backup.", L.YELLOW))
        return False
    if not _acquire_lock(cfg):
        print(L.colorize("git backup skipped: lock busy", L.YELLOW))
        return False
    try:
        branch = _setup_remote(cfg, root)
        _force_add_artifacts(cfg, root)
        r = _run(["git", "commit", "-m",
                  msg or f"backup: training artifacts {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}"],
                 root, check=False)
        if r.returncode != 0 and "nothing to commit" not in (r.stdout or "") + (r.stderr or ""):
            # nothing changed -> fine
            if "no changes" in (r.stdout or "") + (r.stderr or "") or \
               "nothing to commit" in (r.stdout or "") + (r.stderr or ""):
                return False
        p = _run(["git", "push", "origin", branch], root, check=False)
        if p.returncode == 0:
            print(L.colorize("✔ git backup pushed", L.GREEN))
            return True
        print(L.colorize(f"git push failed: {p.stderr}", L.RED))
        # restore clean remote URL (don't leave a bad token embedded in it)
        repo_url = gb.get("repo_url", "")
        if repo_url:
            _run(["git", "remote", "set-url", "origin", repo_url],
                 root, check=False)
        return False
    except Exception as e:
        print(L.colorize(f"git backup error: {e}", L.RED))
        return False
    finally:
        _release_lock(cfg)
