# Cleaning log

Every cleaning step, with counts before and after, and the reason.

## 1. File inventory (`src/data_audit.py`)

| Check | Result |
|---|---|
| Patient files (.txt) | 942 |
| Recordings (.wav / .hea / .tsv) | 3163 / 3163 / 3163 |
| Recordings listed in the .txt files | 3163 |
| Orphan files (missing partner or not listed) | 0 |
| Repeated locations (e.g. `49748_AV_1`) | 43 recordings |

Nothing removed. The files are complete.

## 2. Audio audit (`src/data_audit.py`)

| Flag | Recordings | Action |
|---|---|---|
| Unreadable | 0 | - |
| Empty | 0 | - |
| Silent (RMS < 1e-4) | 0 | - |
| Very short (< 5 s) | 0 | - |
| Wrong sampling rate (not 4000 Hz) | 0 | - |
| .hea and .wav disagree | 0 | - |
| Clipped (> 1% of samples at full scale) | 20 | Kept. Spike removal in preprocessing handles them. |

Durations: 5.15 s to 64.51 s (median 21.46 s). The range 4.8-80.4 s in the
slides is wrong for the public data.

Note: the median peak is 1.0, so most files seem scaled to full range.
That is why 1976 recordings have at least one sample at full scale. One
sample is not a problem, so we only flag files above 1%.

Before: 3163 recordings. After: 3163 recordings.

## 3. Patient table (`src/patient_table.py`)

| Step | Before | After | Reason |
|---|---|---|---|
| Parse the .txt files | 942 files | 942 patients (visits) | One row per visit |
| Structural `nan` in murmur fields | 10031 `nan` | 0 `nan` (now "not applicable") | No murmur means nothing to describe. It is not missing data. |
| Real missing values | - | Age 74, Height 116, Weight 105 | Kept as missing. Not model inputs. |
| Height and Weight to numbers | text | float | - |
| Outliers (outside Q1 - 3 IQR, Q3 + 3 IQR per age group) | - | 7 flagged, 0 removed | Clear errors (e.g. an adolescent of 58 cm), but these columns are not used by the models |
| Merge the 2 visits of a child (Additional ID) | 942 IDs | 872 child_id | 70 children were seen in 2014 and in 2015 |

Note: 70 of the 74 missing ages are pregnant patients. They are probably
young adults whose age group was not recorded.

Note: 15 of the 70 children with 2 visits have different murmur labels in the 2
visits. We keep the label of each visit (the patient is the unit) and keep
both visits in the same set.

## 4. Segmentation (`src/segmentation.py`)

| Check | Result |
|---|---|
| Mean share of a recording that is annotated | 54.5% (median 51.7%) |
| Recordings with less than 50% annotated | 1529 of 3163 |
| Recordings with no annotation | 1 |
| Mean coverage for Unknown patients | 30.2% (Absent 54.9%, Present 58.9%) |

Decision: we do **not** keep only the annotated regions. Half of the audio
would be lost, and a real screening tool has no annotations. We use the full
recording and drop bad windows by quality.

## 5. Label filter and split (`src/split.py`)

| Step | Before | After | Reason |
|---|---|---|---|
| Remove murmur "Unknown" | 942 patients | 874 patients | Unknown = the expert could not decide (often poor quality). Not a class. |
| Split by child_id, stratified | 816 children | train 571, val 122, test 123 children | No child in two sets |

Patients (visits): train 609, val 135, test 130. Present share: 20.7%, 20.0%, 20.0%.

## 6. Preprocessing and windows (`src/preprocess.py`)

| Step | Before | After | Reason |
|---|---|---|---|
| Recordings of labeled patients | 3163 | 3007 | Recordings of Unknown patients removed |
| Cut into 5 s windows, 50% overlap | 3007 recordings | 23037 windows | Fixed input size |
| Drop silent windows (RMS < 0.3) | 23037 | -210 | Stethoscope not on the skin |
| Drop very noisy windows (RMS > 2.0) | | -37 | Big noise bursts |
| Kept windows | | 22790 | 1 recording loses all its windows, 0 patients lose all their windows |

## Summary

| Level | Raw | After cleaning | Removed | Main reason |
|---|---|---|---|---|
| Patient visits | 942 | 874 | 68 | Murmur "Unknown" |
| Children | 872 | 816 | 56 | Only Unknown visits |
| Recordings | 3163 | 3006 | 157 | 156 from Unknown patients, 1 with no good window |
| 5 s windows | 23037 | 22790 | 247 | Silent (210) or very noisy (37) |
| Height/Weight outliers | 7 | 7 (flagged) | 0 | Not model inputs |
