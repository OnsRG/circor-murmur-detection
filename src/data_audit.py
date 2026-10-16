"""Steps 1-2: file inventory and audio audit.
this is the health check of your data, before any cleaning. It answers two questions:

    - Do I have all the files? (Step 1, inventory)
    - Are the audio files usable? (Step 2, audit)
Run:   python src/data_audit.py
Saves: outputs/file_inventory.csv, outputs/orphan_files.csv,
       outputs/audit_recordings.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from utils import get_dir, load_config


def inventory(data_dir: Path) -> pd.DataFrame:
    """List every file in the data folder with its extension and patient ID."""
    rows = []
    for f in sorted(data_dir.iterdir()):
        if f.is_file():
            rows.append({"file": f.name, "ext": f.suffix.lower(), "stem": f.stem,
                         "patient_id": f.stem.split("_")[0]})
    return pd.DataFrame(rows)


def listed_recordings(data_dir: Path) -> pd.DataFrame:
    """Read the header of each patient .txt and return the recordings it lists.

    The first line is "ID n_recordings sample_rate", then one line per
    recording: "location file.hea file.wav file.tsv".
    """
    rows = []
    for txt in sorted(data_dir.glob("*.txt")):
        lines = txt.read_text().splitlines()
        patient_id, n_rec, _ = lines[0].split()
        for line in lines[1:1 + int(n_rec)]:
            location, _, wav, _ = line.split()
            rows.append({"recording": Path(wav).stem, "patient_id": patient_id,
                         "location": location})
    return pd.DataFrame(rows)


def find_orphans(inv: pd.DataFrame, listed: pd.DataFrame) -> pd.DataFrame:
    """Find recordings with a missing .wav/.hea/.tsv or not listed in any .txt."""
    stems = {ext: set(inv.loc[inv.ext == ext, "stem"]) for ext in [".wav", ".hea", ".tsv"]}
    listed_set = set(listed.recording)
    problems = []
    for stem in sorted(set.union(*stems.values()) | listed_set):
        missing = [ext for ext, s in stems.items() if stem not in s]
        if missing:
            problems.append({"recording": stem, "problem": "missing " + " ".join(missing)})
        if stem not in listed_set:
            problems.append({"recording": stem, "problem": "not listed in a patient .txt"})
    return pd.DataFrame(problems, columns=["recording", "problem"])


def read_hea(path: Path) -> tuple[int, int, int]:
    """Return (sample rate, n samples, n channels) from the first line of a .hea file.

    First line format: "name n_signals sample_rate n_samples".
    """
    parts = path.read_text().split()
    return int(float(parts[2])), int(parts[3]), int(parts[1])


def audit_recording(data_dir: Path, rec: str, cfg: dict) -> dict:
    """Read one recording (.hea and .wav), compare them and flag problems."""
    row = {"recording": rec, "unreadable": False, "error": ""}
    try:
        hea_sr, hea_n, hea_ch = read_hea(data_dir / f"{rec}.hea")
        x, sr = sf.read(data_dir / f"{rec}.wav", dtype="float32", always_2d=True)
    except Exception as e:  # missing or broken file
        row.update(unreadable=True, error=str(e))
        return row

    n, ch = x.shape
    sig = x[:, 0]
    row.update(
        sr=sr, n_samples=n, channels=ch, duration_s=n / sr,
        hea_sr=hea_sr, hea_n_samples=hea_n, hea_channels=hea_ch,
        header_match=(sr == hea_sr and n == hea_n and ch == hea_ch),
        rms=float(np.sqrt(np.mean(sig ** 2))) if n else 0.0,
        peak=float(np.max(np.abs(sig))) if n else 0.0,
        # full scale for 16-bit audio read as float is about +-1.0
        clip_fraction=float(np.mean(np.abs(sig) >= 0.999)) if n else 0.0,
    )
    a = cfg["audit"]
    row.update(
        empty=n == 0,
        silent=row["rms"] < a["silent_rms"],
        very_short=row["duration_s"] < a["min_duration_s"],
        clipped=row["clip_fraction"] > a["clip_fraction"],
        wrong_sr=sr != cfg["audio"]["original_sr"],
    )
    return row


def main() -> None:
    cfg = load_config()
    data_dir = get_dir(cfg, "data_dir")
    out_dir = get_dir(cfg, "output_dir", create=True)

    # Step 1: inventory
    inv = inventory(data_dir)
    listed = listed_recordings(data_dir)
    orphans = find_orphans(inv, listed)
    print("Files by type:\n", inv.ext.value_counts().to_string())
    print(f"Patients (.txt): {(inv.ext == '.txt').sum()}")
    print(f"Recordings listed in .txt files: {len(listed)}")
    print(f"Orphan problems: {len(orphans)}")
    for ext in [".txt", ".wav", ".hea", ".tsv"]:
        print(f"Examples {ext}: {inv.loc[inv.ext == ext, 'file'].head(3).tolist()}")
    numbered = listed[listed.recording.str.count("_") > 1]
    print(f"Recordings with a repeated location (e.g. ID_MV_1): {len(numbered)}, "
          f"examples: {numbered.recording.head(3).tolist()}")

    # Step 2: audio audit
    audit = pd.DataFrame([audit_recording(data_dir, r, cfg) for r in listed.recording])
    audit = listed.merge(audit, on="recording")
    flags = ["unreadable", "empty", "silent", "very_short", "clipped", "wrong_sr"]
    print("\nAudio audit")
    print(f"Header and wav agree: {int(audit.header_match.sum())} / {len(audit)}")
    for flag in flags:
        print(f"  {flag:<11}: {int(audit[flag].fillna(False).astype(bool).sum())}")
    print(f"Any clipped sample: {int((audit.clip_fraction > 0).sum())} recordings")
    print("Duration (s):\n", audit.duration_s.describe().round(2).to_string())

    inv.to_csv(out_dir / "file_inventory.csv", index=False)
    orphans.to_csv(out_dir / "orphan_files.csv", index=False)
    audit.to_csv(out_dir / "audit_recordings.csv", index=False)
    print(f"\nSaved 3 CSV files to {out_dir}")


if __name__ == "__main__":
    main()
