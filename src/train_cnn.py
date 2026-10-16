"""Step 10: small CNN on log-mel spectrograms (PyTorch).

- Input normalization: one mean and one std computed on TRAIN windows only.
- Imbalance: positive class weight = n_absent / n_present (train windows).
- Early stopping on the patient-level AUROC of VALIDATION (patience 5).

Run:   python src/train_cnn.py
Saves: outputs/cnn.pt, outputs/preds_cnn.csv, outputs/cnn_history.csv,
       figures/cnn_training.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_baseline import patient_auroc, save_preds
from utils import get_dir, load_config, set_seed

EPOCHS, PATIENCE, BATCH, LR = 30, 5, 64, 1e-3


class SmallCNN(nn.Module):
    """3 conv blocks (16, 32, 64 filters) + global average pooling + 1 output."""

    def __init__(self):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                                 nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)  # one logit per window


def predict(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Return the probability of 'Present' for every window."""
    model.eval()
    probs = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(X), batch_size=256):
            probs.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(probs)


def plot_history(hist: pd.DataFrame, path) -> None:
    """Plot train loss and validation AUROC per epoch."""
    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    ax1.plot(hist.epoch, hist.train_loss, "o-", label="train loss")
    ax2 = ax1.twinx()
    ax2.plot(hist.epoch, hist.val_auroc, "s-", color="tab:orange", label="val AUROC")
    ax1.set(xlabel="epoch", ylabel="train loss", title="CNN training")
    ax2.set_ylabel("val patient AUROC")
    fig.legend(loc="lower center", ncol=2)
    plt.tight_layout(rect=(0, 0.08, 1, 1))  # leave room for the legend
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    out_dir = get_dir(cfg, "output_dir")
    fig_dir = get_dir(cfg, "figures_dir", create=True)

    X = np.load(out_dir / "logmel.npy").astype(np.float32)
    meta = pd.read_csv(out_dir / "features_meta.csv", dtype={"patient_id": str, "child_id": str})
    train = (meta.split == "train").values
    y = meta.label.values.astype(np.float32)

    mean, std = X[train].mean(), X[train].std()  # TRAIN only
    X = torch.from_numpy((X - mean) / std).unsqueeze(1)  # shape (N, 1, mels, frames)
    loader = DataLoader(TensorDataset(X[train], torch.tensor(y[train])),
                        batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg["seed"]))

    model = SmallCNN()
    pos_weight = torch.tensor((y[train] == 0).sum() / (y[train] == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    print(f"Train windows: {train.sum()}, pos_weight: {pos_weight:.2f}")

    history, best_auc, wait = [], -1.0, 0
    val_rows = (meta.split == "val").values
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        val_prob = predict(model, X[val_rows])
        val_auc = patient_auroc(meta[val_rows], val_prob, "val")
        history.append({"epoch": epoch, "train_loss": total / train.sum(), "val_auroc": val_auc})
        print(f"epoch {epoch:2d}  train loss {total / train.sum():.4f}  val patient AUROC {val_auc:.3f}")
        if val_auc > best_auc:
            best_auc, wait = val_auc, 0
            torch.save(model.state_dict(), out_dir / "cnn.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping. Best val AUROC {best_auc:.3f}")
                break

    model.load_state_dict(torch.load(out_dir / "cnn.pt"))
    save_preds(meta, predict(model, X), out_dir / "preds_cnn.csv")
    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / "cnn_history.csv", index=False)

    plot_history(hist, fig_dir / "cnn_training.png")
    print(f"Saved model, predictions and history to {out_dir}")


if __name__ == "__main__":
    main()
