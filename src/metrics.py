"""Evaluation metrics: accuracy, precision, recall, F1 (macro), confusion matrix."""
import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def collect_preds(model, loader, device):
    """Return (logits_sum_loss, y_true, y_pred) over a loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, ys, ps = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        ps.append(logits.argmax(1).cpu())
        ys.append(y.cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    return total_loss / len(y_true), y_true, y_pred


def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def metrics_from_cm(cm):
    """Compute accuracy, precision, recall, F1 (macro) from a confusion matrix."""
    n_classes = cm.shape[0]
    acc = np.trace(cm) / cm.sum()
    pre, rec, f1 = [], [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        pre.append(tp / (tp + fp) if tp + fp > 0 else 0.0)
        rec.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
    for p, r in zip(pre, rec):
        f1.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return {
        "acc": float(acc),
        "precision": float(np.mean(pre)),
        "recall": float(np.mean(rec)),
        "f1": float(np.mean(f1)),
    }


def plot_confusion_matrix(cm, class_names, path, title="Confusion Matrix",
                          normalize=False):
    """Save confusion matrix as PNG using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    thresh = data.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val}"
            if normalize:
                txt = f"{val}\n({data[i, j]:.1%})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="white" if data[i, j] > thresh else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
