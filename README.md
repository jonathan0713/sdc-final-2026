# SDC 2026 Spring Final Competition - Tracking

本專案完成兩個 4D radar multi-object tracking 競賽的 starter code TODO，將每幀 moving cluster 轉成 cluster-level detection，再用 tracker 維持跨幀一致的 tracking ID，最後輸出 Kaggle 需要的 `result.csv`。

## 已完成內容

- `calculate_cluster_centroids()`: 將 `mask_cluster` 中的 moving cluster 轉成 centroid、平均 range rate、cluster points。
- `build_detections()`: 建立 tracker 使用的 `[x, y, z, cluster_id, range_rate]` detection array。
- `Tracker.match_tracks()`: 使用 Hungarian assignment 做 track-to-detection association，並以 centroid distance gating 過濾不合理配對。
- `Tracker.update()`: 加入簡單 constant-velocity prediction、track age 管理、新 track 建立、刪除過久未匹配 track。
- `generate_cluster_track_dict()`: 將本幀 confirmed tracks 回填成 `{cluster_id: track_id}`，供 CSV writer 產生 point-level tracking mask。
- `visualize_tracking()`: 支援可選雷達點與 track trajectory 視覺化。
- `evaluate_tracking.py`: 改成使用標準 `csv` 輸出，避免本機 pandas/NumPy binary ABI 不相容。

## 依賴套件

兩個 competition 的 `starter_code/requirements.txt` 皆為：

```bash
numpy
matplotlib
scipy
```

`matplotlib` 只會在使用 `--plot-every` 時載入；一般產生 `result.csv` 不需要載入視覺化套件。

## 執行方式

### Competition 1: Static Ego

```bash
cd competition_1_static_mot_student/starter_code

python3 main.py --data-root ../public_dev --seq seq_1 --output-root ../outputs
python3 evaluate_tracking.py \
  --gt ../public_dev/seq_1/gt_answer_seq1.csv \
  --pred ../outputs/<seq_1_output>/result.csv \
  --out ../outputs/seq_1_eval.csv

python3 main.py --data-root ../test_input --seq seq_2 --output-root ../outputs
python3 check_result_format.py \
  --submission ../outputs/<seq_2_output>/result.csv \
  --data-root ../test_input \
  --seq seq_2
```

目前已產生可提交檔案：

```text
competition_1_static_mot_student/outputs/seq_2_student_20260602_162718/result.csv
```

### Competition 2: Moving Ego

```bash
cd competition_2_moving_ego_mot_student/starter_code

python3 main.py --data-root ../public_dev --seq seq_3 --output-root ../outputs
python3 evaluate_tracking.py \
  --gt ../public_dev/seq_3/gt_answer_seq3.csv \
  --pred ../outputs/<seq_3_output>/result.csv \
  --out ../outputs/seq_3_eval.csv

python3 main.py --data-root ../test_input --seq seq_4 --output-root ../outputs
python3 check_result_format.py \
  --submission ../outputs/<seq_4_output>/result.csv \
  --data-root ../test_input \
  --seq seq_4
```

目前已產生可提交檔案：

```text
competition_2_moving_ego_mot_student/outputs/seq_4_student_20260602_162724/result.csv
```

## 本機驗證結果

| Competition | Dev Seq | Format | MOTA-style score | FP | FN | IDSW proxy |
|---|---:|---|---:|---:|---:|---:|
| Static Ego | `seq_1` | OK | 0.984788 | 0 | 0 | 153 |
| Moving Ego | `seq_3` | OK | 0.989988 | 0 | 0 | 97 |

測試集格式檢查：

| Competition | Test Seq | Rows | Result |
|---|---:|---:|---|
| Static Ego | `seq_2` | 1284 | OK |
| Moving Ego | `seq_4` | 2400 | OK |

## 參數

預設 tracker 參數：

```text
max_age = 5
min_hits = 1
max_distance = 5.0
```

`min_hits=1` 是為了避免 moving cluster 在剛出現時被標成 `-2` 而造成 missed detection。若想降低可能的 false positive，可提高 `--min-hits`，但 dev score 目前以 `1` 表現較穩。
