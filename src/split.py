"""Step 6: split the data by child_id into train / validation / test.

Rules:
- Only patients with a Present or Absent murmur label (Unknown is excluded).
- Both visits of one child go to the same set (no leakage).
- The split is stratified on the child label: Present if any visit is Present.

Run:   python src/split.py
Saves: outputs/split.csv  (one row per patient visit)
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import get_dir, load_config


def make_split(patients: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return the labeled patients with a 'split' column (train / val / test)."""
    df = patients[patients.Murmur.isin(["Present", "Absent"])].copy()
    df["label"] = (df.Murmur == "Present").astype(int)

    # One row per child: label = 1 if any visit has a murmur
    children = df.groupby("child_id")["label"].max().reset_index()
    s = cfg["split"]
    train, rest = train_test_split(children, train_size=s["train"],
                                   stratify=children.label, random_state=cfg["seed"])
    val, test = train_test_split(rest, test_size=s["test"] / (s["val"] + s["test"]),
                                 stratify=rest.label, random_state=cfg["seed"])

    split_of = {**dict.fromkeys(train.child_id, "train"),
                **dict.fromkeys(val.child_id, "val"),
                **dict.fromkeys(test.child_id, "test")}
    df["split"] = df.child_id.map(split_of)
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    """Stop with an error if a child appears in more than one set."""
    sets_per_child = df.groupby("child_id")["split"].nunique()
    assert sets_per_child.max() == 1, "LEAKAGE: a child is in two sets"
    print("Leakage check passed: every child is in exactly one set.")


def main() -> None:
    cfg = load_config()
    out_dir = get_dir(cfg, "output_dir")
    patients = pd.read_csv(out_dir / "patients.csv",
                           dtype={"Patient ID": str, "child_id": str})

    df = make_split(patients, cfg)
    check_no_leakage(df)

    print(f"Excluded (Unknown murmur): {len(patients) - len(df)} patients")
    summary = df.groupby("split").agg(children=("child_id", "nunique"),
                                      patients=("Patient ID", "size"),
                                      present=("label", "sum"))
    summary["present_%"] = (100 * summary.present / summary.patients).round(1)
    print(summary.loc[["train", "val", "test"]].to_string())

    cols = ["Patient ID", "child_id", "Murmur", "label", "split"]
    df[cols].to_csv(out_dir / "split.csv", index=False)
    print(f"Saved {out_dir / 'split.csv'}")


if __name__ == "__main__":
    main()
