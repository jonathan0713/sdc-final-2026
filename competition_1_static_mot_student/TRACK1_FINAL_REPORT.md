# Track1 Final Report: Static Ego MOT

## Final Submission

Final Track1 version:

```text
outputs/track1_candidate_kalman_reid_md4p5_age2_smoothed_gap25_dist30/result.csv
```

Absolute path:

```text
/home/jonathan/Documents/ROS1/sdc-final-2026/competition_1_static_mot_student/outputs/track1_candidate_kalman_reid_md4p5_age2_smoothed_gap25_dist30/result.csv
```

Final tracker:

```text
tracker_mode = kalman_reid
max_distance = 4.5
max_age = 2
min_hits = 1
postprocess = smooth_tracks.py
smooth max_gap = 25
smooth max_distance = 30
```

Kaggle Track1 result:

```text
Large improvement over the previous best.
```

Final file md5:

```text
e4a1922240e3d7f9701dfa4053af7ca4
```

## Competition Goal

Track1 is a static-ego multi-object tracking task using radar point clouds and moving-object cluster masks. The output is a per-frame `result.csv` that assigns a stable tracking ID to each moving radar point while keeping static points as `-1`.

Because the observer is a vehicle stopped at an intersection, many targets can be close to each other, especially motorcycles and compact vehicles. The hard part is keeping object identity consistent through dense traffic, sparse radar clusters, brief fragmentation, and cluster-ID changes across frames.

## Evaluation Understanding

The local evaluator uses a MOTA-style proxy:

```text
score = 1 - (FP + FN + IDSW) / GT_positive_points
```

In this project, most normal tracking candidates had:

```text
FP = 0
FN = 0
```

So the meaningful target became:

```text
reduce IDSW
```

We avoided exploit-style label tricks and kept the solution as real tracking. The final result is a Kalman + Re-ID tracker followed by an offline tracklet-merge postprocess that connects plausible fragmented tracklets.

## Development Timeline

| Step | Version / Idea | Purpose | Result |
| --- | --- | --- | --- |
| 1 | Baseline cluster tracker | Use cluster centroid matching with simple velocity extrapolation. | Produced valid submissions, but ID stability was limited. |
| 2 | Basic Kalman tracker | Add constant-velocity prediction and Hungarian association. | Improved over the starter behavior, but still had many ID switches in close-object situations. |
| 3 | Kalman Re-ID | Keep deleted/lost tracks and allow new detections to recover old IDs. | Became the strongest core tracker for Track1. |
| 4 | Hyperparameter sweep | Tune `max_distance`, `max_age`, and `min_hits` on public-dev `seq_1`. | Best core setting became `kalman_reid md4.5 age2 hits1`. |
| 5 | Aggressive `max_distance` / `max_age` tests | Check whether larger distance or longer survival helps the static-intersection setting. | Too aggressive settings did not consistently help; the bottleneck was not simple age/distance tuning. |
| 6 | Error analysis | Analyze ID switches and fragmentation around dense vehicle/motorcycle regions. | Human visual debugging was difficult; local metric and tracklet structure suggested postprocess merging was more useful. |
| 7 | Low-point / motorcycle-aware variants | Try to handle sparse radar clusters with special low-point logic. | No reliable improvement over the best Re-ID core tracker. |
| 8 | Feature Re-ID | Add cluster-level feature cost and last-measured-position Re-ID. | Technically reasonable, but did not beat `kalman_reid md4.5 age2` for Track1. |
| 9 | Range-rate Kalman (`kalman_rr`) | Try the Track2-winning range-rate Kalman method on Track1. | Track1 became worse; `kalman_rr` was rejected for Track1. |
| 10 | Conservative smoothing | Merge fragmented tracklets using motion continuity. | `gap20_dist9` matched the downloaded Kaggle best file and improved over pure Re-ID. |
| 11 | Wider smoothing gate | Increase smoothing distance/gap from the best branch. | `gap25_dist10` and `gap25_dist14` improved on Kaggle. |
| 12 | Final smoothing expansion | Continue the same direction to `gap25_dist30`. | User-reported Kaggle result had a large improvement; this became the final Track1 version. |

## Important Candidate Results

### Pure Re-ID Core

```text
outputs/track1_candidate_kalman_reid_md4p5_age2/seq_2_student_20260609_000628/result.csv
```

Public-dev effect:

```text
kalman_reid md4.5 age2: IDSW=36, score=0.996421
```

Conclusion: this was the strongest core tracker and became the base for final postprocessing.

### Previous Kaggle Best

Downloaded best file:

```text
/home/jonathan/Downloads/result.csv
```

Matched local file:

```text
outputs/track1_candidate_kalman_reid_md4p5_age2_smoothed_gap20_dist9/result.csv
```

md5:

```text
558e33648de3f9dd98667ecbb9b8789c
```

Public-dev effect:

```text
pure Re-ID:      IDSW=36, score=0.996421
gap20_dist9:     IDSW=30, score=0.997017
```

Conclusion: smoothing was not just noise; it reduced ID switches by merging plausible fragmented tracklets.

### Gap 25 / Distance 10 and 14

```text
outputs/track1_candidate_kalman_reid_md4p5_age2_smoothed_gap25_dist10/result.csv
outputs/track1_candidate_kalman_reid_md4p5_age2_smoothed_gap25_dist14/result.csv
```

Effect:

```text
Both improved on Kaggle compared with the previous best.
```

Conclusion: hidden data also supported widening the smoothing gate.

### Final Gap 25 / Distance 30

```text
outputs/track1_candidate_kalman_reid_md4p5_age2_smoothed_gap25_dist30/result.csv
```

Public-dev effect:

```text
gap20_dist9:      IDSW=30, score=0.997017
gap25_dist20:     IDSW=25, score=0.997514
gap25_dist30:     IDSW=18, score=0.998210
```

Kaggle effect:

```text
Large improvement.
```

Conclusion: the final Track1 bottleneck was track fragmentation rather than the core frame-to-frame association. A wider offline merge gate recovered many IDs without hurting the observed hidden score.

## Final Method Details

The final pipeline has two stages.

Stage 1: online tracker

- Constant-velocity Kalman tracker.
- Hungarian assignment.
- Re-ID buffer for recently deleted tracks.
- Core parameters:

```text
tracker_mode = kalman_reid
max_distance = 4.5
max_age = 2
min_hits = 1
```

Stage 2: offline smoothing

- Read the generated `result.csv`.
- Build tracklets from per-frame track IDs.
- Estimate each tracklet's velocity from recent observations.
- For each target tracklet, search for a previous source tracklet.
- Merge when:

```text
frame gap <= 25
predicted distance <= 30
target start point count <= 3
source length >= 2
```

The final smoothing produced:

```text
tracklets: 44
candidate_pairs: 26
merged_segments: 26
```

This is a postprocess merge of fragmented tracklets, not a single-ID or metric exploit. It preserves the tracking structure and connects segments that are plausible under motion continuity.

## Reproduce Final Result

Run from:

```bash
cd /home/jonathan/Documents/ROS1/sdc-final-2026/competition_1_static_mot_student/starter_code
```

Stage 1: generate the base Re-ID result.

```bash
uv run --active python main.py \
  --data-root ../test_input \
  --seq seq_2 \
  --output-root ../outputs \
  --tracker-mode kalman_reid \
  --max-distance 4.5 \
  --max-age 2 \
  --min-hits 1
```

Stage 2: apply final smoothing.

```bash
uv run --active python smooth_tracks.py \
  --input ../outputs/track1_candidate_kalman_reid_md4p5_age2/seq_2_student_20260609_000628/result.csv \
  --output ../outputs/track1_candidate_kalman_reid_md4p5_age2_smoothed_gap25_dist30/result.csv \
  --merge-report ../outputs/track1_candidate_kalman_reid_md4p5_age2_smoothed_gap25_dist30/merge_pairs.csv \
  --max-gap 25 \
  --max-distance 30
```

Format check:

```bash
uv run --active python check_result_format.py \
  --submission ../outputs/track1_candidate_kalman_reid_md4p5_age2_smoothed_gap25_dist30/result.csv \
  --data-root ../test_input \
  --seq seq_2
```

Expected format result:

```text
[OK] submission format is valid
sequence: seq_2
frames: 321
rows: 1284
```

## Final Decision

Use:

```text
/home/jonathan/Documents/ROS1/sdc-final-2026/competition_1_static_mot_student/outputs/track1_candidate_kalman_reid_md4p5_age2_smoothed_gap25_dist30/result.csv
```

Do not replace this with Track2's `kalman_rr`. Track1 and Track2 respond differently: `kalman_rr` is excellent for Track2, but Track1's best solution is `kalman_reid md4.5 age2` plus widened offline smoothing.
