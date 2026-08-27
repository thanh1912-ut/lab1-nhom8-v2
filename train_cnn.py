"""Entry: train CNN benchmark. Usage:
    python train_cnn.py                      # all pending runs
    python train_cnn.py --only adam          # only one optimizer
    python train_cnn.py --config configs/cnn.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from benchmark import load_config, run_all  # noqa: E402
from train import get_device, run_id, test_best_weight  # noqa: E402
from gitbackup import backup  # noqa: E402

DEFAULT_CFG = str(Path(__file__).parent / "configs" / "cnn.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--only", default=None,
                    help="run only this optimizer (e.g. adam)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = run_all(cfg, only=args.only, device=get_device())
    if cfg.get("git_backup", {}).get("enabled", False):
        backup(cfg, msg="backup: benchmark complete (cnn)")


if __name__ == "__main__":
    main()
