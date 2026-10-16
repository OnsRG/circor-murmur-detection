"""Step 8: compute model inputs for every kept window.

(A) classical features per window (for the baseline):
    13 MFCC (mean + std), spectral centroid / bandwidth / rolloff / flatness,
    zero-crossing rate (mean + std), energy share in 3 frequency bands.
(B) log-mel spectrogram per window (for the CNN), shape (n_mels, frames).

Run:   python src/features.py
Saves: outputs/features_classical.npy, outputs/logmel.npy,
       outputs/features_meta.csv (same row order as the .npy files)
"""
import librosa
import numpy as np
import pandas as pd

from preprocess import clean_recording, make_windows
from utils import get_dir, load_config

BANDS = [(25, 100), (100, 200), (200, 400)]  # Hz


def classical_features(w: np.ndarray, sr: int, f: dict, fmin: float, fmax: float) -> np.ndarray:
    """Return one vector of hand-made features for one window."""
    spec = dict(sr=sr, n_fft=f["n_fft"], hop_length=f["hop"])
    mfcc = librosa.feature.mfcc(y=w, n_mfcc=f["n_mfcc"], n_mels=f["n_mels"],
                                fmin=fmin, fmax=fmax, **spec)
    frame_feats = np.vstack([
        librosa.feature.spectral_centroid(y=w, **spec),
        librosa.feature.spectral_bandwidth(y=w, **spec),
        librosa.feature.spectral_rolloff(y=w, **spec),
        librosa.feature.spectral_flatness(y=w, n_fft=f["n_fft"], hop_length=f["hop"]),
        librosa.feature.zero_crossing_rate(y=w, frame_length=f["n_fft"], hop_length=f["hop"]),
    ])
    power = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(len(w), 1 / sr)
    band_share = [power[(freqs >= lo) & (freqs < hi)].sum() / power.sum() for lo, hi in BANDS]
    return np.concatenate([mfcc.mean(1), mfcc.std(1),
                           frame_feats.mean(1), frame_feats.std(1), band_share])


def feature_names(f: dict) -> list:
    """Names of the classical features, in the same order as the vector."""
    frame = ["centroid", "bandwidth", "rolloff", "flatness", "zcr"]
    return ([f"mfcc{i}_mean" for i in range(f["n_mfcc"])] + [f"mfcc{i}_std" for i in range(f["n_mfcc"])]
            + [f"{n}_mean" for n in frame] + [f"{n}_std" for n in frame]
            + [f"energy_{lo}_{hi}Hz" for lo, hi in BANDS])


def log_mel(w: np.ndarray, sr: int, f: dict, fmin: float, fmax: float) -> np.ndarray:
    """Return the log-mel spectrogram (dB) of one window."""
    mel = librosa.feature.melspectrogram(y=w, sr=sr, n_fft=f["n_fft"], hop_length=f["hop"],
                                         n_mels=f["n_mels"], fmin=fmin, fmax=fmax)
    return librosa.power_to_db(mel, ref=1.0)


def main() -> None:
    cfg = load_config()
    data_dir = get_dir(cfg, "data_dir")
    out_dir = get_dir(cfg, "output_dir")
    a, w, f = cfg["audio"], cfg["windows"], cfg["features"]
    sr, fmin, fmax = a["target_sr"], a["bandpass_low"], a["bandpass_high"]

    meta = pd.read_csv(out_dir / "windows.csv", dtype={"patient_id": str, "child_id": str})
    meta = meta[meta.kept].reset_index(drop=True)
    index = {wid: i for i, wid in enumerate(meta.window_id)}

    n_frames = int(w["length_s"] * sr) // f["hop"] + 1
    X_cls = np.zeros((len(meta), len(feature_names(f))), dtype=np.float32)
    X_mel = np.zeros((len(meta), f["n_mels"], n_frames), dtype=np.float16)  # float16 saves disk

    recordings = meta.recording.unique()
    for k, rec in enumerate(recordings):
        windows = make_windows(clean_recording(data_dir / f"{rec}.wav", cfg),
                               sr, w["length_s"], w["overlap"])
        for i, win in enumerate(windows):
            row = index.get(f"{rec}_w{i:02d}")
            if row is None:  # dropped window (silent or noisy)
                continue
            X_cls[row] = classical_features(win, sr, f, fmin, fmax)
            X_mel[row] = log_mel(win, sr, f, fmin, fmax)
        if (k + 1) % 500 == 0:
            print(f"  {k + 1} / {len(recordings)} recordings")

    np.save(out_dir / "features_classical.npy", X_cls)
    np.save(out_dir / "logmel.npy", X_mel)
    meta.to_csv(out_dir / "features_meta.csv", index=False)
    print(f"Classical features: {X_cls.shape}, log-mel: {X_mel.shape}")
    print(f"NaN in features: {np.isnan(X_cls).sum()}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
