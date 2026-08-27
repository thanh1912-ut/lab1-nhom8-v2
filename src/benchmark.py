"""Benchmark orchestration: scan yaml config -> run all (optimizer, lr, seed)
combos, track results.json, test best weights, trigger git backup."""
import argparse
import json
import os

try:
    from . import logger as L
    from .gitbackup import backup
    from .train import (get_device, run_id, test_best_weight, train_one_run)
except ImportError:
    import logger as L
    from gitbackup import backup
    from train import (get_device, run_id, test_best_weight, train_one_run)


def load_config(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def results_path(cfg):
    """Per-model results file so MLP/CNN processes can run in parallel
    without racing on a shared results.json."""
    return os.path.join(cfg["output_dir"], f"results_{cfg['model']}.json")


def all_results_paths(cfg):
    """All results files in the output dir (results_mlp.json, results_cnn.json,
    plus legacy results.json)."""
    out = cfg["output_dir"]
    paths = []
    if os.path.isdir(out):
        for f in sorted(os.listdir(out)):
            if f.startswith("results") and f.endswith(".json"):
                paths.append(os.path.join(out, f))
    return paths


def load_results(cfg):
    p = results_path(cfg)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(cfg, results):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(results_path(cfg), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_all(cfg, only=None, device=None):
    """Run every (optimizer, lr, seed) combo; skip finished runs.
    only: optional optimizer name filter."""
    device = device or get_device()
    results = load_results(cfg)
    model = cfg["model"]
    combos = [(o, lr, s) for o, spec in cfg["optimizers"].items()
              for lr in spec["lrs"] for s in cfg.get("seed_runs", [42])]
    if only:
        combos = [c for c in combos if c[0] == only]
    done = sum(1 for k, h in results.items() if h.get("done"))
    print(f"{L.colorize('Benchmark', L.BOLD)} model={model}: {done} done, "
          f"{len(combos) - sum(1 for c in combos if results.get(run_id(model, *c), {}).get('done'))} pending. "
          f"Device: {device}")
    n_done_this_session = 0
    backup_every = int(cfg.get("git_backup", {}).get("every_runs", 1))
    for opt, lr, seed in combos:
        rid = run_id(model, opt, lr, seed)
        if results.get(rid, {}).get("done"):
            continue
        h = train_one_run(cfg, model, opt, lr, seed, device=device)
        results[rid] = h
        if h.get("done"):
            t = test_best_weight(cfg, h, device=device)
            if t:
                print(f"  {L.colorize('test', L.BOLD)} acc {t['acc']:.4f} "
                      f"pre {t['precision']:.4f} rec {t['recall']:.4f} "
                      f"f1 {t['f1']:.4f}")
        save_results(cfg, results)
        n_done_this_session += 1
        if (cfg.get("git_backup", {}).get("enabled", False)
                and n_done_this_session % backup_every == 0):
            backup(cfg, results_path(cfg))
    return results
