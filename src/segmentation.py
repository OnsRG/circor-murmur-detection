"""Step 5: read the .tsv segmentation files and measure their coverage.

Each .tsv row is: start (s), end (s), label
(0 = unannotated, 1 = S1, 2 = systole, 3 = S2, 4 = diastole).
This file checks how much of each recording has expert labels
Run:   python src/segmentation.py
Saves: outputs/segmentation_coverage.csv, figures/segmentation_example.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from utils import get_dir, load_config

NAMES = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}
COLORS = {1: "tab:red", 2: "tab:orange", 3: "tab:blue", 4: "tab:green"}


def read_tsv(path) -> pd.DataFrame:
    """Read one segmentation file as a table with start, end, label."""
    return pd.read_csv(path, sep="\t", header=None, names=["start", "end", "label"])


def coverage(seg: pd.DataFrame, duration: float) -> dict:
    """Share of the recording that is annotated, and the number of beats (S1)."""
    ann = seg[seg.label > 0]
    covered = (ann.end - ann.start).sum()
    return {"annotated_s": covered,
            "coverage": covered / duration if duration else 0.0,
            "n_beats": int((seg.label == 1).sum())}


def plot_example(data_dir, rec: str, path, seconds: float = 4.0) -> None:
    """Plot the waveform with S1, systole, S2 and diastole colored."""
    x, sr = sf.read(data_dir / f"{rec}.wav")
    seg = read_tsv(data_dir / f"{rec}.tsv")
    start = seg[seg.label > 0].start.min()  # begin at the first annotated part
    x = x[int(start * sr): int((start + seconds) * sr)]
    t = start + np.arange(len(x)) / sr
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(t, x, color="black", lw=0.5)
    for _, s in seg[(seg.label > 0) & (seg.end > start) & (seg.start < start + seconds)].iterrows():
        ax.axvspan(s.start, s.end, color=COLORS[s.label], alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in COLORS.values()]
    ax.legend(handles, NAMES.values(), loc="upper right", ncol=4)
    ax.set_xlim(start, start + seconds)
    ax.set_xlabel("time (s)")
    ax.set_title(f"Segmentation of {rec}")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    cfg = load_config()
    data_dir = get_dir(cfg, "data_dir")
    out_dir = get_dir(cfg, "output_dir")
    fig_dir = get_dir(cfg, "figures_dir", create=True)

    audit = pd.read_csv(out_dir / "audit_recordings.csv", dtype={"patient_id": str})
    patients = pd.read_csv(out_dir / "patients.csv", dtype={"Patient ID": str})
    rows = []
    for rec, dur in zip(audit.recording, audit.duration_s):
        rows.append({"recording": rec, "duration_s": dur,
                     **coverage(read_tsv(data_dir / f"{rec}.tsv"), dur)})
    cov = pd.DataFrame(rows).merge(audit[["recording", "patient_id"]], on="recording")
    cov = cov.merge(patients[["Patient ID", "Murmur"]], left_on="patient_id",
                    right_on="Patient ID").drop(columns="Patient ID")

    print("Coverage (share of each recording that is annotated):")
    print(cov.coverage.describe().round(3).to_string())
    print(f"Recordings with no annotation at all: {(cov.coverage == 0).sum()}")
    print(f"Recordings with coverage < 50%: {(cov.coverage < 0.5).sum()}")
    print("Mean coverage by murmur label:\n",
          cov.groupby("Murmur").coverage.mean().round(3).to_string())
    print(f"Median beats per recording: {cov.n_beats.median():.0f}")

    # Example: most audible location of a loud holosystolic murmur, well annotated
    loud = patients[(patients["Systolic murmur grading"] == "III/VI")
                    & (patients["Systolic murmur timing"] == "Holosystolic")]
    recs = loud["Patient ID"] + "_" + loud["Most audible location"]
    best = cov[cov.recording.isin(recs) & (cov.coverage > 0.9)].iloc[0].recording
    plot_example(data_dir, best, fig_dir / "segmentation_example.png")
    cov.to_csv(out_dir / "segmentation_coverage.csv", index=False)
    print(f"Saved {out_dir / 'segmentation_coverage.csv'} and the example plot ({best})")


if __name__ == "__main__":
    main()
