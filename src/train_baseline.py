"""Step 9: classical baseline on window features + a dummy model.

- Dummy: always says Absent (probability 0). This is the bar to beat.
- Logistic regression and random forest, both with class weights.
  The scaler is fitted on TRAIN only (inside the pipeline).
- We keep the model with the best patient-level AUROC on VALIDATION.

Run:   python src/train_baseline.py
Saves: outputs/preds_dummy.csv, outputs/preds_baseline.csv, outputs/baseline.joblib
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import to_patient
from utils import get_dir, load_config, set_seed


def patient_auroc(meta: pd.DataFrame, prob: np.ndarray, split: str) -> float:
    """AUROC after averaging window probabilities per patient."""
    pat = to_patient(meta.assign(prob=prob), "mean")
    pat = pat[pat.split == split]
    return roc_auc_score(pat.label, pat.prob)


def save_preds(meta: pd.DataFrame, prob: np.ndarray, path) -> None:
    """Save window probabilities for all sets (test is only scored in evaluate.py)."""
    cols = ["window_id", "patient_id", "child_id", "split", "label"]
    meta[cols].assign(prob=prob).to_csv(path, index=False)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    out_dir = get_dir(cfg, "output_dir")

    X = np.load(out_dir / "features_classical.npy")
    meta = pd.read_csv(out_dir / "features_meta.csv", dtype={"patient_id": str, "child_id": str})
    train = (meta.split == "train").values
    y = meta.label.values
    print(f"Train windows: {train.sum()} ({y[train].mean():.1%} Present)")

    models = {
        "logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "random forest": make_pipeline(
            StandardScaler(), RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                n_jobs=-1, random_state=cfg["seed"])),
    }
    best_name, best_auc, best_prob = None, -1.0, None
    for name, model in models.items():
        model.fit(X[train], y[train])
        prob = model.predict_proba(X)[:, 1]
        val = (meta.split == "val").values
        win_auc = roc_auc_score(y[val], prob[val])
        pat_auc = patient_auroc(meta, prob, "val")
        print(f"{name:<20} val AUROC: window {win_auc:.3f}, patient {pat_auc:.3f}")
        if pat_auc > best_auc:
            best_name, best_auc, best_prob, best_model = name, pat_auc, prob, model

    print(f"Chosen baseline (best validation AUROC): {best_name}")
    save_preds(meta, best_prob, out_dir / "preds_baseline.csv")
    save_preds(meta, np.zeros(len(meta)), out_dir / "preds_dummy.csv")
    joblib.dump(best_model, out_dir / "baseline.joblib")
    print(f"Saved predictions and model to {out_dir}")


if __name__ == "__main__":
    main()
