"""Training engine: early stopping, periodic validation, batch-level resume,
TensorBoard, metrics (acc/pre/rec/F1) and confusion matrices."""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

_TB_OK_CACHE = None


def _tensorboard_works():
    """EventFileWriter's native binary segfaults on some setups (e.g. py3.13
    + anaconda). Probe it in a subprocess so a crash can't kill this process.
    NOTE: torch.utils.tensorboard must only be imported lazily AFTER the
    probe passes - importing it at module top crashes the whole process."""
    global _TB_OK_CACHE
    if _TB_OK_CACHE is not None:
        return _TB_OK_CACHE
    if os.environ.get("LAB1_DISABLE_TB"):
        _TB_OK_CACHE = False
        return False
    import subprocess
    import sys
    code = ("from tensorboard.summary.writer.event_file_writer import "
            "EventFileWriter as E; import tempfile; "
            "w=E(tempfile.mkdtemp()); w.close()")
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, timeout=60)
        _TB_OK_CACHE = (r.returncode == 0)
    except Exception:
        _TB_OK_CACHE = False
    if not _TB_OK_CACHE:
        print("TensorBoard disabled on this machine (native writer crash). "
              "Metrics are still logged to train.log / results.json.")
    return _TB_OK_CACHE


def _make_summary_writer(tb_dir):
    try:
        if not _tensorboard_works():
            return None
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(tb_dir)
    except KeyboardInterrupt:
        raise
    except Exception:
        return None

try:
    from . import metrics as M
    from . import logger as L
    from .data import load_datasets, make_train_loader, make_eval_loader
    from .models import MODELS, count_params
except ImportError:
    import metrics as M
    import logger as L
    from data import load_datasets, make_train_loader, make_eval_loader
    from models import MODELS, count_params

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "rmsprop": torch.optim.RMSprop,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "adamax": torch.optim.Adamax,
    "nadam": torch.optim.NAdam,
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_id(model, opt, lr, seed):
    return f"{model}_{opt}_lr{lr}_s{seed}"


def gpu_mem_str(device):
    if device.type == "cuda":
        return f"{torch.cuda.memory_reserved() / 1e9:.2f}G"
    return "-"


def train_one_run(cfg, model_name, opt_name, lr, seed, device=None,
                  resume=True, max_batches=None):
    """Train one (model, optimizer, lr, seed) run.

    - Validates every cfg val_every epochs: loss + acc/pre/rec/F1 + confusion
      matrix PNG + TensorBoard scalars.
    - Early stop when val loss has not improved for cfg early_stop_patience
      epochs (checked at each validation).
    - best.pt = lowest val loss; last.pt = end of run.
    - Checkpoint every cfg checkpoint_every_batches batches + each epoch end;
      resume restores epoch/batch + RNG states.
    Returns history dict.
    """
    device = device or get_device()
    rid = run_id(model_name, opt_name, lr, seed)
    out_root = cfg["output_dir"]
    run_dir = os.path.join(out_root, "runs", rid)
    tb_dir = os.path.join(cfg.get("tensorboard_dir", out_root + "/tensorboard"), rid)
    os.makedirs(run_dir, exist_ok=True)

    epochs = int(cfg["epochs"])
    val_every = int(cfg.get("val_every", 5))
    patience = int(cfg.get("early_stop_patience", 30))
    ckpt_every = int(cfg.get("checkpoint_every_batches", 200))
    batch_size = int(cfg["data"]["batch_size"])
    sel = cfg.get("selection", {}) or {}
    best_by = str(sel.get("best_weight_by", "loss")).lower()   # loss | acc | f1
    stop_by = str(sel.get("early_stop_by", "loss")).lower()    # loss | acc | f1
    if best_by not in ("loss", "acc", "f1"):
        best_by = "loss"
    if stop_by not in ("loss", "acc", "f1"):
        stop_by = "loss"

    flogger = L.get_file_logger(rid, os.path.join(run_dir, "train.log"))
    L.print_run_header(f"run {rid}", [
        ("model", model_name),
        ("optimizer", f"{opt_name} lr={lr}"),
        ("seed", seed),
        ("device", str(device)),
        ("epochs", f"{epochs} (val every {val_every}, patience {patience})"),
        ("select", f"best_weight_by={best_by}, early_stop_by={stop_by}"),
        ("params", f"{count_params(MODELS[model_name]()):,}"),
    ])
    L.log_event(flogger, f"START {rid} opt={opt_name} lr={lr} seed={seed} "
                         f"device={device} epochs={epochs} "
                         f"best_by={best_by} stop_by={stop_by}")

    set_seed(seed)
    model = MODELS[model_name]().to(device)
    optimizer = _OPTIMIZERS[opt_name](model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "run_id": rid, "model": model_name, "optimizer": opt_name,
        "lr": lr, "seed": seed, "device": str(device),
        "selection": {"best_weight_by": best_by, "early_stop_by": stop_by},
        "epochs": [], "train_loss": [], "val_epoch": [],
        "val_loss": [], "val_acc": [], "val_precision": [],
        "val_recall": [], "val_f1": [],
        "best_epoch": None, "best_val_loss": None,
        "best_val_metric": None, "best_val_metric_name": best_by,
        "stopped_early": False, "done": False,
        "train_time_sec": 0.0,
    }
    best_val_loss = float("inf")      # cho resume + history
    best_sel_value = -float("inf")    # metric được chọn (acc/f1); loss xử lý riêng
    if best_by == "loss":
        best_sel_value = float("inf")
    stop_sel_value = -float("inf")
    if stop_by == "loss":
        stop_sel_value = float("inf")
    epochs_since_improve = 0
    start_epoch, start_batch = 0, 0

    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        history = ckpt["history"]
        best_val_loss = ckpt["best_val_loss"]
        best_sel_value = ckpt.get("best_sel_value", best_val_loss)
        stop_sel_value = ckpt.get("stop_sel_value", best_val_loss)
        epochs_since_improve = ckpt.get("epochs_since_improve", 0)
        start_epoch, start_batch = ckpt["epoch"], ckpt["batch_idx"]
        random.setstate(ckpt["py_rng"])
        np.random.set_state(ckpt["np_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt.get("cuda_rng"))
        msg = f"RESUMED from epoch {start_epoch}, batch {start_batch}"
        L.log_event(flogger, msg)
        print(L.colorize(f"↻ {rid}: {msg}", L.YELLOW))

    def save_ckpt(epoch, batch_idx):
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        torch.save({
            "epoch": epoch, "batch_idx": batch_idx,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "history": history, "best_val_loss": best_val_loss,
            "best_sel_value": best_sel_value,
            "stop_sel_value": stop_sel_value,
            "epochs_since_improve": epochs_since_improve,
            "torch_rng": torch.get_rng_state(),
            "np_rng": np.random.get_state(),
            "py_rng": random.getstate(),
            "cuda_rng": cuda_rng,
        }, ckpt_path)

    train_ds, val_ds, _ = load_datasets(cfg)
    val_loader = make_eval_loader(val_ds, cfg["data"].get("val_batch_size", 512))

    writer = None

    def validate(epoch):
        nonlocal best_val_loss, epochs_since_improve, best_sel_value, \
            stop_sel_value
        val_loss, y_true, y_pred = M.collect_preds(model, val_loader, device)
        cm = M.confusion_matrix(y_true, y_pred)
        m = M.metrics_from_cm(cm)
        history["val_epoch"].append(epoch + 1)
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(m["acc"], 6))
        history["val_precision"].append(round(m["precision"], 6))
        history["val_recall"].append(round(m["recall"], 6))
        history["val_f1"].append(round(m["f1"], 6))
        if writer:
            step = (epoch + 1)
            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/acc", m["acc"], step)
            writer.add_scalar("val/precision", m["precision"], step)
            writer.add_scalar("val/recall", m["recall"], step)
            writer.add_scalar("val/f1", m["f1"], step)
        # selection: compare on the configured metric (loss: lower is better;
        # acc/f1: higher is better)
        if best_by == "loss":
            best_now = val_loss
            improved = best_now < best_sel_value
        else:
            best_now = m[best_by]
            improved = best_now > best_sel_value
        if stop_by == "loss":
            stop_now = val_loss
            stop_improved = stop_now < stop_sel_value
        else:
            stop_now = m[stop_by]
            stop_improved = stop_now > stop_sel_value
        mark = L.colorize(" ← best", L.GREEN) if improved else ""
        L.print_val_metrics(epoch + 1, m, best_mark=mark)
        L.log_event(flogger, f"EPOCH {epoch + 1} VAL loss={val_loss:.6f} "
                             f"acc={m['acc']:.4f} pre={m['precision']:.4f} "
                             f"rec={m['recall']:.4f} f1={m['f1']:.4f}"
                             f"{' BEST' if improved else ''}")
        if improved:
            best_sel_value = best_now
            best_val_loss = val_loss
            history["best_epoch"] = epoch + 1
            history["best_val_loss"] = round(val_loss, 6)
            history["best_val_metric"] = round(best_now, 6)
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
            cm_path = os.path.join(run_dir, f"cm_best.png")
            M.plot_confusion_matrix(
                cm, CLASS_NAMES, cm_path,
                title=f"{rid} best val (epoch {epoch + 1}, {best_by})")
        if stop_improved:
            stop_sel_value = stop_now
            epochs_since_improve = 0
        else:
            epochs_since_improve += val_every
        return improved

    prev_time = float(history.get("train_time_sec", 0.0))
    t0 = time.time()
    cur_epoch, cur_batch = start_epoch, start_batch
    interrupted = False
    stop_early = False
    try:
        writer = _make_summary_writer(tb_dir)
        for epoch in range(start_epoch, epochs):
            cur_epoch, cur_batch = epoch, 0
            loader = make_train_loader(train_ds, epoch, batch_size, seed)
            model.train()
            running_loss, running_n = 0.0, 0
            nb = max_batches if max_batches is not None else len(loader)
            for bi, (x, y) in enumerate(loader):
                if epoch == start_epoch and bi < start_batch:
                    continue
                if bi >= nb:
                    break
                cur_batch = bi
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * y.size(0)
                running_n += y.size(0)
                if (bi + 1) % ckpt_every == 0:
                    history["train_time_sec"] = round(
                        prev_time + time.time() - t0, 2)
                    save_ckpt(epoch, bi + 1)
                if writer and ((bi + 1) % 100 == 0):
                    writer.add_scalar("train/loss",
                                      running_loss / max(running_n, 1),
                                      epoch * len(loader) + bi + 1)
            train_loss = running_loss / max(running_n, 1)
            history["epochs"].append(epoch + 1)
            history["train_loss"].append(round(train_loss, 6))
            history["train_time_sec"] = round(prev_time + time.time() - t0, 2)
            if writer:
                writer.add_scalar("train/loss_epoch", train_loss, epoch + 1)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"],
                                  epoch + 1)
            last = (epoch + 1 == epochs)
            L.print_epoch_line(epoch + 1, epochs, train_loss,
                               f"{history['train_time_sec']:.0f}s", last=last)
            L.log_event(flogger, f"EPOCH {epoch + 1} train_loss={train_loss:.6f}")
            torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))
            if (epoch + 1) % val_every == 0 or last:
                validate(epoch)
            save_ckpt(epoch + 1, 0)
            if epochs_since_improve >= patience:
                stop_early = True
                break
    except KeyboardInterrupt:
        interrupted = True
        save_ckpt(cur_epoch, cur_batch)
        L.print_interrupted(rid, cur_epoch, cur_batch)
        L.log_event(flogger, f"INTERRUPTED epoch={cur_epoch} batch={cur_batch}")
    finally:
        if writer:
            writer.close()

    if not interrupted:
        history["stopped_early"] = stop_early
        history["done"] = True
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        L.log_event(flogger, f"DONE best_epoch={history['best_epoch']} "
                             f"best_val_loss={history['best_val_loss']} "
                             f"early_stop={stop_early} "
                             f"time={history['train_time_sec']}s")
    return history


def test_best_weight(cfg, h, device=None):
    """Load best.pt of a run, evaluate on the test set with full metrics.
    Saves test metrics into history dict and confusion matrix PNG."""
    device = device or get_device()
    rid = h["run_id"]
    run_dir = os.path.join(cfg["output_dir"], "runs", rid)
    best_path = os.path.join(run_dir, "best.pt")
    if not os.path.exists(best_path):
        return None
    model = MODELS[h["model"]]()
    model.load_state_dict(torch.load(best_path, map_location="cpu",
                                      weights_only=True))
    model.to(device)
    _, _, test_ds = load_datasets(cfg)
    test_loader = make_eval_loader(test_ds, cfg["data"].get("test_batch_size", 512))
    _, y_true, y_pred = M.collect_preds(model, test_loader, device)
    cm = M.confusion_matrix(y_true, y_pred)
    m = M.metrics_from_cm(cm)
    h["test"] = {k: round(v, 6) for k, v in m.items()}
    M.plot_confusion_matrix(cm, CLASS_NAMES,
                            os.path.join(run_dir, "cm_test.png"),
                            title=f"{rid} test (best weight)")
    L.log_event(None, "") if False else None
    return h["test"]
