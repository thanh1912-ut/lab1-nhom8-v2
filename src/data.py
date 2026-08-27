"""FashionMNIST data loading and train/val/test split."""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SPLIT_SEED = 42


def load_datasets(cfg):
    """Return (train_ds, val_ds, test_ds) from config dict."""
    d = cfg["data"]
    transform = transforms.ToTensor()
    train_full = datasets.FashionMNIST(d["root"], train=True, download=True,
                                       transform=transform)
    test_ds = datasets.FashionMNIST(d["root"], train=False, download=True,
                                    transform=transform)
    n_val = int(d.get("val_split", 6000))
    n_train = len(train_full) - n_val
    train_ds, val_ds = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(SPLIT_SEED))
    return train_ds, val_ds, test_ds


def make_train_loader(train_ds, epoch, batch_size, seed):
    """Deterministic per-epoch shuffle so batch-level resume is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed * 10000 + epoch)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      generator=g)


def make_eval_loader(ds, batch_size):
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
