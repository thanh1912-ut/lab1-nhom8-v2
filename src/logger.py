"""YOLO-style (ultralytics-like) console + file logger."""
import logging
import os
import sys

# ANSI colors
R = "\033[0m"       # reset
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"

IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(color, text):
    return f"{color}{text}{R}" if IS_TTY else str(text)


def colorize(text, color):
    return _c(color, text)


def print_run_header(title, rows, width=64):
    """Print an ultralytics-style info box, e.g.:
    ╭───────────────────────────╮
    │ run: mlp_adam_lr0.001_s42 │
    ╰───────────────────────────╯
    """
    lines = [f"{title}:"] + [f"  {k:<14}{v}" for k, v in rows]
    inner = max(len(line) for line in lines) + 2
    print(_c(CYAN, "╭" + "─" * (inner + 2) + "╮"))
    for line in lines:
        pad = inner - len(line)
        print(_c(CYAN, "│") + f" {line}{' ' * pad} " + _c(CYAN, "│"))
    print(_c(CYAN, "╰" + "─" * (inner + 2) + "╯"))
    sys.stdout.flush()


def print_epoch_line(epoch, epochs, train_loss, elapsed, last=False):
    """One compact line per epoch, ultralytics style."""
    tag = _c(GREY, "      ┃ ") if not last else _c(GREY, "      ┗━ ")
    body = (f"{_c(BOLD, f'{epoch:>3d}')}{_c(GREY, f'/{epochs}')} "
            f"{_c(GREY, 'Epoch')} "
            f"{_c(GREY, 'GPU_mem:')} {_c(YELLOW, 'n/a')} "
            f"{_c(GREY, 'box_loss:')} {_c(GREEN, f'{train_loss:8.4f}')} "
            f"{_c(GREY, 'elapsed:')} {_c(YELLOW, elapsed)}")
    print(tag + body)
    sys.stdout.flush()


def print_val_metrics(epoch, m, best_mark=""):
    """Print validation metrics table row (called every val_every epochs)."""
    header = (_c(BOLD, "                 Class"))
    header += _c(GREY, f"  {'Images':>7}{'Instances':>10}")
    header += _c(GREY, f"{'acc':>8}{'pre':>8}{'rec':>8}{'f1':>8}")
    print(_c(GREY, "      ┃ ") + header)
    row = (_c(BOLD, "                 all"))
    row += _c(GREY, f"  {6000:>7}{6000:>10}")
    row += (f"{m['acc']:>8.4f}{m['precision']:>8.4f}"
            f"{m['recall']:>8.4f}{m['f1']:>8.4f} {best_mark}")
    print(_c(GREY, "      ┃ ") + row)
    sys.stdout.flush()


def print_run_done(run_id, best_epoch, best_val_loss, test_m, time_sec,
                   stopped_early=False):
    status = _c(YELLOW, "early-stopped") if stopped_early else _c(GREEN, "done")
    print()
    print(f"{_c(BOLD, '✔')} {run_id} {status} in {time_sec:.1f}s. "
          f"Best val epoch {_c(BOLD, str(best_epoch))} "
          f"(val loss {best_val_loss:.4f}).")
    if test_m:
        print(f"  Test: acc {test_m['acc']:.4f} | pre {test_m['precision']:.4f} "
              f"| rec {test_m['recall']:.4f} | f1 {test_m['f1']:.4f}")
    print(_c(GREY, "─" * 64))
    sys.stdout.flush()


def print_interrupted(run_id, epoch, batch):
    print(f"\n{_c(RED, '✋ Interrupted')} {run_id} at epoch {epoch + 1}, "
          f"batch {batch}. Checkpoint saved - rerun to resume.")
    sys.stdout.flush()


def get_file_logger(name, path):
    """Plain-text file logger (no ANSI codes) for tail -f."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    return logger


def log_event(logger, msg):
    if logger:
        logger.info(msg)


def strip_ansi(s):
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)
