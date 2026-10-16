"""Step 7: clean the audio and cut it into fixed windows.

Pipeline for one recording:
  resample 4000 -> 2000 Hz -> band-pass 25-400 Hz -> spike removal
  -> normalize (divide by the recording's own std) -> 5 s windows, 50% overlap
  -> drop silent or very noisy windows.

The normalization uses only the recording itself, so it cannot leak
information between sets. Statistics that are learned from many recordings
(feature scaling) are fitted on TRAIN only, in the training scripts.

Run:   python src/preprocess.py
Saves: outputs/windows.csv (one row per window, kept or dropped),
       figures/preprocess_before_after.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

from utils import get_dir, load_config


def bandpass(x: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    """4th-order Butterworth band-pass, run forward and backward (no phase shift)."""
    sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


def remove_spikes(x: np.ndarray, sr: int, factor: float = 3.0) -> np.ndarray:
    """Clip large spikes (bumps, rubbing).

    We cut the signal in 0.5 s blocks and take the peak of each block. Normal
    heart sounds are close to the median peak. Anything above `factor` times
    the median peak is clipped to that limit.
    """
    block = sr // 2
    n = len(x) // block
    peaks = np.abs(x[: n * block]).reshape(n, block).max(axis=1)
    limit = factor * np.median(peaks)
    return np.clip(x, -limit, limit)


def clean_recording(path, cfg: dict) -> np.ndarray:
    """Load one .wav file and apply resample, band-pass, spike removal, normalization."""
    a = cfg["audio"]
    x, sr = sf.read(path, dtype="float32")
    x = resample_poly(x, a["target_sr"], sr)  # 4000 -> 2000 Hz
    x = bandpass(x, a["target_sr"], a["bandpass_low"], a["bandpass_high"])
    x = remove_spikes(x, a["target_sr"])
    return (x / (x.std() + 1e-8)).astype(np.float32)


def make_windows(x: np.ndarray, sr: int, length_s: float, overlap: float) -> np.ndarray:
    """Cut a signal into fixed windows. Pad with zeros if shorter than one window."""
    size = int(length_s * sr)
    hop = int(size * (1 - overlap))
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    starts = range(0, len(x) - size + 1, hop)  # the last incomplete part is cropped
    return np.stack([x[s: s + size] for s in starts])


def window_flags(windows: np.ndarray, cfg: dict) -> pd.DataFrame:
    """RMS of each window and whether it is silent or very noisy.

    The recording has std 1 after normalization, so a window with
    RMS << 1 is (almost) silent and a window with RMS >> 1 holds a big burst
    of noise (crying, speech, rubbing).
    """
    w = cfg["windows"]
    rms = np.sqrt((windows ** 2).mean(axis=1))
    return pd.DataFrame({"rms": rms,
                         "silent": rms < w["silent_rms"],
                         "noisy": rms > w["noisy_rms"]})


def plot_before_after(data_dir, recs: list, cfg: dict, path) -> None:
    """Plot the raw and cleaned signal (first 8 s) for a few recordings."""
    fig, axes = plt.subplots(len(recs), 2, figsize=(13, 2.6 * len(recs)))
    for row, rec in enumerate(recs):
        raw, sr = sf.read(data_dir / f"{rec}.wav")
        clean = clean_recording(data_dir / f"{rec}.wav", cfg)
        new_sr = cfg["audio"]["target_sr"]
        axes[row, 0].plot(np.arange(8 * sr) / sr, raw[: 8 * sr], lw=0.4)
        axes[row, 1].plot(np.arange(8 * new_sr) / new_sr, clean[: 8 * new_sr], lw=0.4, color="tab:green")
        axes[row, 0].set_title(f"{rec}: raw (4000 Hz)")
        axes[row, 1].set_title(f"{rec}: cleaned (2000 Hz, 25-400 Hz, spikes clipped, std = 1)")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    cfg = load_config()
    data_dir = get_dir(cfg, "data_dir")
    out_dir = get_dir(cfg, "output_dir")
    fig_dir = get_dir(cfg, "figures_dir", create=True)
    sr, w = cfg["audio"]["target_sr"], cfg["windows"]

    split = pd.read_csv(out_dir / "split.csv", dtype={"Patient ID": str, "child_id": str})
    audit = pd.read_csv(out_dir / "audit_recordings.csv", dtype={"patient_id": str})
    recs = audit.merge(split, left_on="patient_id", right_on="Patient ID")
    print(f"Recordings of labeled patients: {len(recs)}")

    rows = []
    for r in recs.itertuples():
        windows = make_windows(clean_recording(data_dir / f"{r.recording}.wav", cfg),
                               sr, w["length_s"], w["overlap"])
        flags = window_flags(windows, cfg)
        for i, f in flags.iterrows():
            rows.append({"window_id": f"{r.recording}_w{i:02d}", "recording": r.recording,
                         "patient_id": r.patient_id, "child_id": r.child_id,
                         "location": r.location, "label": r.label, "split": r.split,
                         "start_s": i * w["length_s"] * (1 - w["overlap"]),
                         "rms": f.rms, "silent": f.silent, "noisy": f.noisy})
    win = pd.DataFrame(rows)
    win["kept"] = ~(win.silent | win.noisy)

    print(f"Windows: {len(win)}, silent: {win.silent.sum()}, noisy: {win.noisy.sum()}, "
          f"kept: {win.kept.sum()}")
    lost = set(win.recording) - set(win[win.kept].recording)
    lost_pat = set(win.patient_id) - set(win[win.kept].patient_id)
    print(f"Recordings with no kept window: {len(lost)}, patients with no kept window: {len(lost_pat)}")
    print("Kept windows per split and label:")
    print(win[win.kept].groupby(["split", "label"]).size().unstack().to_string())

    noisy_rec = audit.sort_values("clip_fraction").iloc[-1].recording
    plot_before_after(data_dir, [recs.recording.iloc[0], noisy_rec], cfg,
                      fig_dir / "preprocess_before_after.png")
    win.to_csv(out_dir / "windows.csv", index=False)
    print(f"Saved {out_dir / 'windows.csv'}")


if __name__ == "__main__":
    main()
