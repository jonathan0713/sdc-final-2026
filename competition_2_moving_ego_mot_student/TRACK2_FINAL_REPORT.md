# Track2 Final Report: Moving Ego MOT

## Final Submission

Final Track2 version:

```text
outputs/seq_4_student_20260609_184234/result.csv
```

Absolute path:

```text
/home/jonathan/Documents/ROS1/sdc-final-2026/competition_2_moving_ego_mot_student/outputs/seq_4_student_20260609_184234/result.csv
```

Final tracker:

```text
tracker_mode = kalman_rr
max_distance = 4.0
max_age = 10
min_hits = 1
ego_mode = none
```

Kaggle Track2 result: `1.00`

## Competition Goal

Track2 is a moving-ego multi-object tracking task using radar point clouds and moving-object cluster masks. The output is a per-frame `result.csv` that assigns a stable tracking ID to each moving radar point while keeping static points as `-1`.

The main scoring pressure comes from ID consistency. In our public-dev evaluation, FP and FN were usually zero because the provided moving/static mask already identifies moving radar points. The practical bottleneck was therefore reducing ID switches when nearby clusters cross, disappear briefly, or reappear.

## Evaluation Understanding

The local evaluator uses a MOTA-style proxy:

```text
score = 1 - (FP + FN + IDSW) / GT_positive_points
```

In this project, most candidates had:

```text
FP = 0
FN = 0
```

So the meaningful debugging target became:

```text
reduce IDSW
```

We avoided exploit-style label tricks and kept the solution as real tracking: prediction, gating, assignment, lifecycle handling, and Re-ID experiments.

## Development Timeline

| Step | Version / Idea | Purpose | Result |
| --- | --- | --- | --- |
| 1 | Baseline cluster tracker | Use cluster centroid matching with simple velocity extrapolation. | Worked as a starter, but Track2 public-dev IDSW was worse than later Kalman/Re-ID versions. |
| 2 | Basic Kalman tracker | Add constant-velocity prediction and Mahalanobis-style association. | Improved tracking stability compared with naive matching. |
| 3 | Re-ID Kalman tracker | Keep deleted/lost tracks for a short time and allow later detections to recover old IDs. | Became the strongest early Track2 version. User-reported Kaggle result improved over earlier Kalman variants. |
| 4 | Ego translation compensation experiments | Try `add_xy` and related ego-coordinate offsets using `ego_global_pos.txt`. | `track2_candidate_kalman_ego_add_xy_md5_age8` became worse, so explicit ego translation compensation was abandoned. |
| 5 | Segment split / smoothing postprocess | Merge short fragmented tracklets when the motion continuity is plausible. | `segment_split4_gap20_dist9` slightly improved Track2 hidden score. |
| 6 | Cost threshold variants | Add stricter merge filtering such as `cost5` and `cost3`. | `cost5` matched the best segment-split score; `cost3` was lower. |
| 7 | Age tuning | Test `age5` and `age8` around the best Re-ID + smoothing setup. | `age5` did not change score; `age8` became worse. |
| 8 | Low-point / motorcycle-aware variants | Try to help sparse radar clusters, especially close motorcycles or tiny objects. | No clear Track2 improvement over the best segment-split candidate. |
| 9 | Feature Re-ID | Add cluster-level feature cost and use last measured position for lost-track matching. | Public-dev improved slightly: `IDSW 42 -> 41`, but this was not the final hidden winner. |
| 10 | Range-rate Kalman (`kalman_rr`) | Port the other project's radar-only range-rate Kalman idea: radial-velocity initialization, range-rate EKF update, Euclidean + Mahalanobis gate, range-rate/y cost. | Track2 Kaggle reached `1.00`; this became the final version. |

## Important Candidate Results

### Re-ID Candidate

```text
outputs/track2_candidate_kalman_reid_md7_age5/seq_4_student_20260608_235641/result.csv
```

Effect:

```text
Better than earlier Kalman versions.
```

This was the first strong Track2 direction. It showed that preserving old IDs after short disappearances was useful.

### Ego Compensation Candidate

```text
outputs/track2_candidate_kalman_ego_add_xy_md5_age8/seq_4_student_20260608_235104/result.csv
```

Effect:

```text
Worse than the previous best.
```

Conclusion: external ego translation compensation was not reliable for this pipeline. The radar detections and range-rate association were better handled in the original tracking frame.

### Segment Split Candidate

```text
outputs/track2_candidate_kalman_reid_md7_age5_segment_split4_gap20_dist9/result.csv
```

Effect:

```text
Slight improvement over the previous best.
```

Conclusion: conservative offline tracklet merging helped when a true object was briefly fragmented.

### Segment Cost Candidate

```text
outputs/track2_candidate_kalman_reid_md7_age5_segment_split4_gap20_dist9_cost5/result.csv
```

Effect:

```text
Same score as segment_split4.
```

Conclusion: cost filtering was safe but did not add extra hidden-score gain.

### Feature Re-ID Candidate

```text
outputs/track2_candidate_kalman_feature_reid_md7_age8_segment_split4_gap20_dist9_cost5/result.csv
```

Public-dev effect:

```text
kalman_reid         md=7 age=8: IDSW=42, score=0.995665
kalman_feature_reid md=7 age=8: IDSW=41, score=0.995768
```

Conclusion: cluster feature cost and measured-position Re-ID were technically sound, but the final hidden Track2 result was beaten by `kalman_rr`.

### Final `kalman_rr` Candidate

```text
outputs/seq_4_student_20260609_184234/result.csv
```

Effect:

```text
Kaggle Track2 score: 1.00
```

Conclusion: for Track2, range-rate Kalman was the decisive improvement.

## Final Method Details

The final `kalman_rr` tracker uses:

- Constant-velocity Kalman state:

```text
[x, y, vx, vy]
```

- Detection format:

```text
[x, y, z, cluster_id, range_rate, ...features]
```

- Initial velocity from radar range-rate:

```text
v_init = (range_rate / norm([x, y])) * [x, y]
```

- EKF-style update using:

```text
[x, y, range_rate]
```

- Association gates:

```text
Euclidean hard gate
Mahalanobis soft gate
```

- Matching cost:

```text
mahalanobis_position_cost
+ range_rate_consistency_cost
+ lateral_y_cost
```

This worked especially well for Track2 because the hidden sequence appears compatible with radial-velocity continuity. It also avoided the failed explicit ego-compensation path.

## Reproduce Final Result

Run from:

```bash
cd /home/jonathan/Documents/ROS1/sdc-final-2026/competition_2_moving_ego_mot_student/starter_code
```

Command:

```bash
uv run --active python main.py \
  --data-root ../test_input \
  --seq seq_4 \
  --output-root ../outputs \
  --tracker-mode kalman_rr \
  --max-distance 4.0 \
  --max-age 10 \
  --min-hits 1
```

Format check:

```bash
uv run --active python check_result_format.py \
  --submission ../outputs/seq_4_student_20260609_184234/result.csv \
  --data-root ../test_input \
  --seq seq_4
```

Expected format result:

```text
[OK] submission format is valid
sequence: seq_4
frames: 600
rows: 2400
```

## Final Decision

Use:

```text
/home/jonathan/Documents/ROS1/sdc-final-2026/competition_2_moving_ego_mot_student/outputs/seq_4_student_20260609_184234/result.csv
```

Do not replace it with Track1-tuned settings. Track1 and Track2 respond differently: `kalman_rr` is excellent for Track2 but performed worse on Track1 public-dev, so the final Track2 solution should remain exactly this `kalman_rr` version.
