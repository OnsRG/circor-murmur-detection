"""Step 11: patient-level evaluation of all models.

1. Combine window probabilities into one probability per patient (mean or max).
2. Choose the decision threshold on VALIDATION (best F1). Never on test.
3. Report AUROC, PR-AUC, F1, sensitivity, specificity, accuracy and the
   confusion matrix on validation and test.
4. Error analysis on test: which Present patients are missed, and why.

Run:   python src/evaluate.py   (after train_baseline.py and train_cnn.py)
Saves: reports/results_table.md, outputs/errors_test.csv, figures/eval_*.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

from utils import get_dir, load_config

MODELS = {"dummy": "Dummy (always Absent)", "baseline": "Baseline", "cnn": "Small CNN"}


def to_patient(preds: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """Combine window probabilities into one row per patient (mean or max)."""
    return (preds.groupby(["patient_id", "split"], as_index=False)
                 .agg(label=("label", "first"), prob=("prob", how), n_windows=("prob", "size")))


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Return the threshold with the best F1 on the given (validation) data."""
    grid = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """All metrics for one set of patients at one threshold."""
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    one_class = len(np.unique(p)) == 1  # constant model: AUROC is 0.5 by definition
    return {"AUROC": 0.5 if one_class else roc_auc_score(y, p),
            "PR-AUC": average_precision_score(y, p),
            "F1": f1_score(y, pred, zero_division=0),
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "accuracy": (tp + tn) / len(y), "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def bootstrap_auroc(y: np.ndarray, p: np.ndarray, seed: int, n: int = 1000) -> tuple:
    """95% confidence interval of the AUROC by resampling patients."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            scores.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def plot_curves(results: dict, fig_dir) -> None:
    """ROC and precision-recall curves on test, one line per model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, df in results.items():
        y, p = df.label.values, df.prob.values
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(rec, prec, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC (test, patient level)")
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8)
    axes[1].set(xlabel="recall (sensitivity)", ylabel="precision", title="Precision-recall (test)")
    for ax in axes:
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "eval_curves_test.png", dpi=120)
    plt.close()


def plot_confusion(m: dict, title: str, path) -> None:
    """Draw a 2x2 confusion matrix."""
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Absent", "Present"])
    ax.set_yticks([0, 1], ["Absent", "Present"])
    ax.set(xlabel="predicted", ylabel="true", title=title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    cfg = load_config()
    out_dir = get_dir(cfg, "output_dir")
    fig_dir = get_dir(cfg, "figures_dir", create=True)
    rep_dir = get_dir(cfg, "reports_dir", create=True)

    rows, test_probs, chosen = [], {}, None
    for key, name in MODELS.items():
        preds = pd.read_csv(out_dir / f"preds_{key}.csv", dtype={"patient_id": str})
        for how in ["mean", "max"]:
            pat = to_patient(preds, how)
            val, test = pat[pat.split == "val"], pat[pat.split == "test"]
            thr = choose_threshold(val.label.values, val.prob.values)
            for set_name, df in [("val", val), ("test", test)]:
                rows.append({"model": name, "aggregation": how, "set": set_name,
                             "threshold": thr, **metrics(df.label.values, df.prob.values, thr)})
            # Final model = best validation AUROC (dummy excluded)
            val_auc = rows[-2]["AUROC"]
            if key != "dummy" and (chosen is None or val_auc > chosen["val_auc"]):
                chosen = {"name": f"{name} ({how})", "val_auc": val_auc, "thr": thr, "test": test}
            if how == "mean" and key != "dummy":  # dummy = the dashed chance lines
                test_probs[name] = test

    table = pd.DataFrame(rows)
    show = table.copy()
    num = ["AUROC", "PR-AUC", "F1", "sensitivity", "specificity", "accuracy", "threshold"]
    show[num] = show[num].round(3)
    print(show.to_string(index=False))

    # Final model: confidence interval, plots, error analysis
    test = chosen["test"]
    lo, hi = bootstrap_auroc(test.label.values, test.prob.values, cfg["seed"])
    final = metrics(test.label.values, test.prob.values, chosen["thr"])
    print(f"\nFinal model (chosen on validation AUROC): {chosen['name']}")
    print(f"Test AUROC {final['AUROC']:.3f} (95% CI {lo:.3f}-{hi:.3f}), threshold {chosen['thr']:.2f}")
    plot_curves(test_probs, fig_dir)
    plot_confusion(final, f"{chosen['name']}, test", fig_dir / "eval_confusion_test.png")

    patients = pd.read_csv(out_dir / "patients.csv", dtype={"Patient ID": str})
    cols = ["Patient ID", "Age", "Campaign", "Murmur locations", "Most audible location",
            "Systolic murmur timing", "Systolic murmur grading", "Systolic murmur pitch",
            "Diastolic murmur timing"]
    test = test.merge(patients[cols], left_on="patient_id", right_on="Patient ID")
    test["pred"] = (test.prob >= chosen["thr"]).astype(int)
    test["error"] = np.select([(test.label == 1) & (test.pred == 0), (test.label == 0) & (test.pred == 1)],
                              ["missed (FN)", "false alarm (FP)"], "correct")
    test.drop(columns="Patient ID").to_csv(out_dir / "errors_test.csv", index=False)
    present = test[test.label == 1]
    print("\nTest Present patients, found vs missed, by murmur grade:")
    print(pd.crosstab(present["Systolic murmur grading"], present.error).to_string())
    print("\nBy murmur timing:")
    print(pd.crosstab(present["Systolic murmur timing"], present.error).to_string())
    print("\nMissed patients:")
    print(present[present.error != "correct"][["patient_id", "prob", "Systolic murmur grading",
          "Systolic murmur timing", "Murmur locations", "n_windows"]].round(3).to_string(index=False))

    with open(rep_dir / "results_table.md", "w", encoding="utf-8") as f:
        f.write(show.to_markdown(index=False))
        f.write(f"\n\nFinal model: {chosen['name']}. Test AUROC {final['AUROC']:.3f} "
                f"(95% bootstrap CI {lo:.3f}-{hi:.3f}).\n")
    print(f"\nSaved {rep_dir / 'results_table.md'} and figures")


if __name__ == "__main__":
    main()
