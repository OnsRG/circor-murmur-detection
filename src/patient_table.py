"""Step 3: build one clean patient table from the .txt files.

Run:   python src/patient_table.py
Saves: outputs/patients.csv
"""
import numpy as np
import pandas as pd

from utils import get_dir, load_config

SYSTOLIC = ["Systolic murmur timing", "Systolic murmur shape", "Systolic murmur grading",
            "Systolic murmur pitch", "Systolic murmur quality"]
DIASTOLIC = [c.replace("Systolic", "Diastolic") for c in SYSTOLIC]
MURMUR_FIELDS = ["Murmur locations", "Most audible location"] + SYSTOLIC + DIASTOLIC
NOT_APPLICABLE = "not applicable"  # structural nan: nothing to describe


def parse_txt(path) -> dict:
    """Read one patient .txt: the header line and the '#Key: value' lines."""
    lines = path.read_text().splitlines()
    patient_id, n_rec, _ = lines[0].split()
    row = {"Patient ID": patient_id, "n_recordings": int(n_rec),
           "locations": "+".join(line.split()[0] for line in lines[1:1 + int(n_rec)])}
    for line in lines:
        if line.startswith("#"):
            key, value = line[1:].split(":", 1)
            row[key.strip()] = value.strip()
    return row


def mark_structural_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 'nan' that means 'nothing to describe' by 'not applicable'.

    - No murmur (Absent or Unknown): all murmur fields are not applicable.
    - Murmur Present but no systolic (or diastolic) murmur: those fields are
      not applicable.
    - No second visit: Additional ID is not applicable.
    Every 'nan' left after this is real missing data.
    """
    df = df.replace("nan", np.nan)
    no_murmur = df["Murmur"] != "Present"
    df.loc[no_murmur, MURMUR_FIELDS] = NOT_APPLICABLE
    for fields in (SYSTOLIC, DIASTOLIC):
        empty = df[fields].isna().all(axis=1) & ~no_murmur
        df.loc[empty, fields] = NOT_APPLICABLE
    df["Additional ID"] = df["Additional ID"].fillna(NOT_APPLICABLE)
    return df


def add_child_id(df: pd.DataFrame) -> pd.DataFrame:
    """Give the two visits of the same child (2014 and 2015) one child_id.

    The pairs are mutual (A points to B and B points to A), so the smaller
    ID of the pair is a simple and stable child_id.
    """
    ids = set(df["Patient ID"])
    child = {}
    for pid, other in zip(df["Patient ID"], df["Additional ID"]):
        if other != NOT_APPLICABLE:
            assert other in ids, f"{pid} points to unknown ID {other}"
            child[pid] = min(pid, other, key=int)
    df["child_id"] = df["Patient ID"].map(child).fillna(df["Patient ID"])
    assert df.groupby("child_id").size().max() <= 2, "a child has more than 2 visits"
    return df


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag Height/Weight outside [Q1 - 3*IQR, Q3 + 3*IQR] inside each age group.

    We only flag (we do not delete): these columns are not model inputs.
    """
    for col in ["Height", "Weight"]:
        q1 = df.groupby("Age")[col].transform(lambda s: s.quantile(0.25))
        q3 = df.groupby("Age")[col].transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        df[f"{col}_outlier"] = (df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)
    return df


def main() -> None:
    cfg = load_config()
    data_dir = get_dir(cfg, "data_dir")
    out_dir = get_dir(cfg, "output_dir", create=True)

    df = pd.DataFrame([parse_txt(p) for p in sorted(data_dir.glob("*.txt"))])
    print(f"Patients parsed: {len(df)}")
    raw_nan = (df[MURMUR_FIELDS] == "nan").sum().sum()

    df = mark_structural_nan(df)
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce")
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["Pregnancy status"] = df["Pregnancy status"] == "True"
    df = add_child_id(df)
    df = flag_outliers(df)

    left_nan = df[MURMUR_FIELDS].isna().sum().sum()
    print(f"'nan' in murmur fields: {raw_nan} before, {left_nan} real missing after")
    print("Real missing values:\n", df[["Age", "Height", "Weight"]].isna().sum().to_string())
    print("Age missing vs pregnancy:\n",
          pd.crosstab(df["Age"].fillna("missing"), df["Pregnancy status"]).to_string())
    print(f"Children (child_id): {df.child_id.nunique()} for {len(df)} visits")

    pairs = df[df.duplicated("child_id", keep=False)].groupby("child_id")
    diff = pairs["Murmur"].nunique().gt(1).sum()
    print(f"Children with 2 visits: {pairs.ngroups}, with different murmur labels: {diff}")
    print("Murmur:\n", df["Murmur"].value_counts().to_string())
    out = df[df.Height_outlier | df.Weight_outlier]
    print(f"Height/Weight outliers (flagged, not removed): {len(out)}")
    print(out[["Patient ID", "Age", "Height", "Weight"]].to_string(index=False))

    df.to_csv(out_dir / "patients.csv", index=False)
    print(f"Saved {out_dir / 'patients.csv'}")


if __name__ == "__main__":
    main()
