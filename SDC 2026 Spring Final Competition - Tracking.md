# SDC 2026 Spring Final Competition - Tracking

## Overview

This competition focuses on **4D radar multi-object tracking (MOT)**.

You are given:

- 4D radar point clouds
- pre-computed moving cluster masks
- camera images for visualization
- ego-position files for the moving-ego competition

Your task is to assign **temporally consistent tracking IDs** to moving radar points across frames.

If the same object appears in multiple frames, your algorithm should keep assigning the same track ID to that object.

---

## Competitions

There are two independent competitions.

| Competition | Scenario | Development Sequence | Kaggle Test Sequence |
|---|---|---|---|
| Competition 1 | Static-ego tracking | `seq_1` | `seq_2` |
| Competition 2 | Moving-ego tracking | `seq_3` | `seq_4` |

Development sequences include ground truth and can be used for local evaluation.

Kaggle test sequences do **not** include ground truth. You must generate `result.csv` and submit it to Kaggle.

## Kaggle Link

[SDC_final_competition_Tracking_1 (Static-Ego 4D Radar Multi-Object Tracking)](https://www.kaggle.com/t/253e552e85b447cdae4bd2ff074fe1a8)
[SDC_final_competition_Tracking_2 (Moving-Ego 4D Radar Multi-Object Tracking)](https://www.kaggle.com/t/d70680afb8df4c3cb84f4736147fc6fe)

---

# 1. Competition 1: Static-Ego 4D Radar MOT

## 1.1 Data Package

Download and extract:

```text
competition_1_static_mot_student.zip
```

Expected folder structure:

```text
competition_1_static_mot_student/
├── public_dev/
│   └── seq_1/
│       ├── gt_answer_seq1.csv
│       ├── image/
│       ├── mask_cluster/
│       └── radar/
├── test_input/
│   └── seq_2/
│       ├── image/
│       ├── mask_cluster/
│       └── radar/
├── starter_code/
│   ├── main.py
│   ├── sdc_tracker.py
│   ├── sdc_tracking_utils.py
│   ├── evaluate_tracking.py
│   ├── check_result_format.py
│   ├── README.md
│   ├── TODO_GUIDE.md
│   └── requirements.txt
└── sample_submission_seq2.csv
```

## 1.2 Sequence Usage

| Sequence | Usage | Ground Truth |
|---|---|---|
| `seq_1` | Local development | Available |
| `seq_2` | Kaggle submission | Hidden |

Use `seq_1` to develop and evaluate your algorithm locally.

Use `seq_2` to generate your Kaggle submission.

---

# 2. Competition 2: Moving-Ego 4D Radar MOT

## 2.1 Data Package

Download and extract:

```text
competition_2_moving_ego_mot_student.zip
```

Expected folder structure:

```text
competition_2_moving_ego_mot_student/
├── public_dev/
│   └── seq_3/
│       ├── ego_global_pos.txt
│       ├── gt_answer_seq3.csv
│       ├── image/
│       ├── mask_cluster/
│       └── radar/
├── test_input/
│   └── seq_4/
│       ├── ego_global_pos.txt
│       ├── image/
│       ├── mask_cluster/
│       └── radar/
├── starter_code/
│   ├── main.py
│   ├── sdc_tracker.py
│   ├── sdc_tracking_utils.py
│   ├── evaluate_tracking.py
│   ├── check_result_format.py
│   ├── README.md
│   ├── TODO_GUIDE.md
│   └── requirements.txt
└── sample_submission_seq4.csv
```

## 2.2 Sequence Usage

| Sequence | Usage | Ground Truth | Ego Position |
|---|---|---|---|
| `seq_3` | Local development | Available | Available |
| `seq_4` | Kaggle submission | Hidden | Available |

Use `seq_3` to develop and evaluate your algorithm locally.

Use `seq_4` to generate your Kaggle submission.

## 2.3 Ego-Motion Information

Competition 2 provides:

```text
ego_global_pos.txt
```

You may use this file to design:

- ego-motion compensation
- ego-aware data association
- ego-aware motion prediction
- improved gating under moving-ego conditions

You are not required to use ego information, but it may help improve tracking performance.

---

# 3. Radar Data and Mask Format

Each sequence contains two main data folders:

```text
radar/
mask_cluster/
```

## 3.1 Radar Point Cloud

Each radar frame is stored as a binary file in:

```text
radar/
```

Each radar point has the following format:

```text
[x, y, z, RangeRate, RCS]
```

| Field | Meaning |
|---|---|
| `x` | forward position |
| `y` | lateral position |
| `z` | height |
| `RangeRate` | radial velocity measurement |
| `RCS` | radar cross section |

## 3.2 Cluster Mask

Each frame also has a corresponding mask file in:

```text
mask_cluster/
```

The cluster mask assigns one label to each radar point.

| Mask Value | Meaning |
|---|---|
| `-1` | static point |
| `>= 0` | moving cluster ID |

Example:

```text
-1
-1
0
0
0
1
1
2
2
```

This means:

- points with `-1` are static
- points with `0` belong to moving cluster 0
- points with `1` belong to moving cluster 1
- points with `2` belong to moving cluster 2

Your tracker should assign temporally consistent track IDs to these moving clusters.

---

# 4. Task Definition

The input already provides moving cluster masks.

Your goal is **not** to perform object detection from raw radar.

Your goal is to perform **multi-object tracking**:

```text
moving radar clusters
        ↓
cluster-level detections
        ↓
track association across frames
        ↓
temporally consistent track IDs
        ↓
result.csv
```

For each frame, your algorithm should output a tracking label for every radar point:

| Output Value | Meaning |
|---|---|
| `-1` | static point |
| `-2` | moving point but not assigned to a confirmed track |
| `0, 1, 2, ...` | confirmed track ID |

Note: The label `-2` is only for tentative points. Labeling a true moving object as `-2` counts as a missed detection. (Increase FN)

---

# 5. Submission Format

Your Kaggle submission must be a CSV file named:

```text
result.csv
```

It must contain two columns:

```text
id,mask
```

Each frame must have 4 rows:

```text
frame_id, tracking mask
frame_idx, x coordinates
frame_idy, y coordinates
frame_idz, z coordinates
```

Example:

```text
id,mask
0,-1 -1 0 0 0 1 1 -2 -2
0x,1.2 1.5 3.1 3.2 3.3 4.5 4.6 5.1 5.2
0y,0.1 0.2 1.0 1.1 1.2 2.0 2.1 3.0 3.1
0z,0.0 0.0 0.1 0.1 0.1 0.2 0.2 0.3 0.3
1,-1 -1 0 0 1 1 -2 -2
1x,...
1y,...
1z,...
```

Use the provided sample submission as a reference:

| Competition | Sample Submission |
|---|---|
| Competition 1 | `sample_submission_seq2.csv` |
| Competition 2 | `sample_submission_seq4.csv` |

---

# 6. Starter Code

The starter code is located in:

```text
starter_code/
```

Files:

```text
main.py
sdc_tracker.py
sdc_tracking_utils.py
evaluate_tracking.py
check_result_format.py
README.md
TODO_GUIDE.md
requirements.txt
```

The starter code is a **framework**, not a finished tracking baseline.

---

# 7. Installation

Enter the starter code folder:

```bash
cd starter_code
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The required packages are:

```text
numpy
pandas
matplotlib
scipy
```

---

# 8. Run Starter Code

## 8.1 Competition 1: Run on Development Sequence

From:

```text
competition_1_static_mot_student/starter_code/
```

Run:

```bash
python main.py \
  --data-root ../public_dev \
  --seq seq_1 \
  --output-root ../outputs
```

This will generate an output folder under:

```text
../outputs/
```

Example:

```text
../outputs/seq_1_student_YYYYMMDD_HHMMSS/result.csv
```

## 8.2 Competition 1: Local Evaluation on seq_1

```bash
PRED=$(ls -td ../outputs/seq_1_* | head -1)/result.csv

python evaluate_tracking.py \
  --gt ../public_dev/seq_1/gt_answer_seq1.csv \
  --pred "$PRED" \
  --out ../outputs/seq_1_eval.csv
```

## 8.3 Competition 1: Generate Kaggle Submission for seq_2

```bash
python main.py \
  --data-root ../test_input \
  --seq seq_2 \
  --output-root ../outputs
```

Check submission format:

```bash
PRED=$(ls -td ../outputs/seq_2_* | head -1)/result.csv

python check_result_format.py \
  --submission "$PRED" \
  --data-root ../test_input \
  --seq seq_2
```

Submit this file to Kaggle:

```text
../outputs/seq_2_*/result.csv
```

## 8.4 Competition 2: Run on Development Sequence

From:

```text
competition_2_moving_ego_mot_student/starter_code/
```

Run:

```bash
python main.py \
  --data-root ../public_dev \
  --seq seq_3 \
  --output-root ../outputs
```

## 8.5 Competition 2: Local Evaluation on seq_3

```bash
PRED=$(ls -td ../outputs/seq_3_* | head -1)/result.csv

python evaluate_tracking.py \
  --gt ../public_dev/seq_3/gt_answer_seq3.csv \
  --pred "$PRED" \
  --out ../outputs/seq_3_eval.csv
```

## 8.6 Competition 2: Generate Kaggle Submission for seq_4

```bash
python main.py \
  --data-root ../test_input \
  --seq seq_4 \
  --output-root ../outputs
```

Check submission format:

```bash
PRED=$(ls -td ../outputs/seq_4_* | head -1)/result.csv

python check_result_format.py \
  --submission "$PRED" \
  --data-root ../test_input \
  --seq seq_4
```

Submit this file to Kaggle:

```text
../outputs/seq_4_*/result.csv
```

---

# 9. What You Need to Implement

The starter code contains several TODO sections.

## 9.1 `main.py`

You may need to complete:

| TODO | Purpose |
|---|---|
| `TODO-1` | Compute cluster centroids and cluster-level features |
| `TODO-2` | Map confirmed tracks back to cluster IDs |
| `TODO-3` | Optional visualization |
| `TODO-4` | Build detections for the tracker |
| `TODO-5` | Call tracker update function |

## 9.2 `sdc_tracker.py`

You may need to complete:

| TODO | Purpose |
|---|---|
| `TODO-6` | Implement data association and gating |

A minimum working tracker should include:

1. cluster centroid extraction
2. detection construction
3. detection-to-track association
4. track update
5. new track creation
6. unmatched track handling
7. cluster-to-track output mapping

---

# 10. Evaluation Metric

The competition uses a tracking-oriented score based on:

- false positive points
- false negative points
- ID switch proxy

The score is higher when:

- fewer moving points are missed
- fewer static points are incorrectly tracked
- object IDs remain more stable over time

A simplified form is:

```text
MOTA_proxy = 1 - (FP + FN + IDSW) / GT_positive_points
```

Higher score is better.

---
# 11. Grade
Competition Ranking : 50%
* Public Scoring (20% per competition):
    1. MOTA < Baseline A: 0%
    2. Baseline A ≤ MOTA < Baseline B: 15%
    3. MOTA > Baseline : 20%
* Private Scoring (5% per competition):
Score distributed linearly based on final rank.

Report and Presentation : 50%


---
# 12. Rules

## 12.1 Allowed

You may:

- modify the starter code
- implement your own tracker
- use radar point cloud features
- use cluster masks
- use images for visualization
- use public development ground truth for local evaluation

## 12.2 Not Allowed

You may not:

- manually label the Kaggle test sequences
- submit files that do not follow the required CSV format
- use information from another team's private submission
---



