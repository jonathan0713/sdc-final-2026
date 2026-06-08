# SDC 2026 Spring Final Competition - Tracking

本專案完成兩個 4D radar multi-object tracking 競賽的 starter code TODO，將每幀 moving cluster 轉成 cluster-level detection，再用 tracker 維持跨幀一致的 tracking ID，最後輸出 Kaggle 需要的 `result.csv`。

## 已完成內容

- `calculate_cluster_centroids()`: 將 `mask_cluster` 中的 moving cluster 轉成 centroid、平均 range rate、cluster points。
- `build_detections()`: 建立 tracker 使用的 `[x, y, z, cluster_id, range_rate]` detection array。
- `Tracker.match_tracks()`: 使用 Hungarian assignment 做 track-to-detection association，並以 centroid distance gating 過濾不合理配對。
- `Tracker.update()`: 加入簡單 constant-velocity prediction、track age 管理、新 track 建立、刪除過久未匹配 track。
- `generate_cluster_track_dict()`: 將本幀 confirmed tracks 回填成 `{cluster_id: track_id}`，供 CSV writer 產生 point-level tracking mask。
- `debug_tracking_viewer.py`: 支援彈窗播放 camera image + radar bird's-eye tracking，可暫停、跳格，也可選擇匯出 MP4。
- `evaluate_tracking.py`: 改成使用標準 `csv` 輸出，避免本機 pandas/NumPy binary ABI 不相容。

## 依賴套件

兩個 competition 的 `starter_code/requirements.txt` 皆為：

```bash
numpy
matplotlib
scipy
```

`matplotlib` 只會在使用視覺化 viewer 或 `--plot-every` 時載入；一般產生 `result.csv` 不需要載入視覺化套件。

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
competition_1_static_mot_student/outputs/seq_2_student_20260608_232659/result.csv
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
competition_2_moving_ego_mot_student/outputs/seq_4_student_20260608_232659/result.csv
```

## 本機驗證結果

| Competition | Dev Seq | Format | MOTA-style score | FP | FN | IDSW proxy |
|---|---:|---|---:|---:|---:|---:|
| Static Ego | `seq_1` | OK | 0.988666 | 0 | 0 | 114 |
| Moving Ego | `seq_3` | OK | 0.991226 | 0 | 0 | 85 |

測試集格式檢查：

| Competition | Test Seq | Rows | Result |
|---|---:|---:|---|
| Static Ego | `seq_2` | 1284 | OK |
| Moving Ego | `seq_4` | 2400 | OK |

## 參數

grid search 後的預設 tracker 參數：

| Competition | `max_age` | `min_hits` | `max_distance` |
|---|---:|---:|---:|
| Static Ego | 8 | 1 | 7.0 |
| Moving Ego | 5 | 1 | 7.0 |

`min_hits=1` 是為了避免 moving cluster 在剛出現時被標成 `-2` 而造成 missed detection。若想降低可能的 false positive，可提高 `--min-hits`，但 dev score 目前以 `1` 表現較穩。

## Grid Search

可用 `grid_search.py` 在 public dev 上搜尋 baseline 參數：

```bash
cd competition_1_static_mot_student/starter_code

uv run --active python grid_search.py \
  --data-root ../public_dev \
  --seq seq_1 \
  --output-root ../outputs
```

```bash
cd competition_2_moving_ego_mot_student/starter_code

uv run --active python grid_search.py \
  --data-root ../public_dev \
  --seq seq_3 \
  --output-root ../outputs
```

summary 位置：

```text
competition_1_static_mot_student/outputs/seq_1_grid_search_20260608_232546/grid_search_summary.csv
competition_2_moving_ego_mot_student/outputs/seq_3_grid_search_20260608_232606/grid_search_summary.csv
```

## 互動式 Debug Viewer

產生 `result.csv` 後，可以用彈窗播放對齊的相機影像與雷達 tracking。

```bash
cd competition_1_static_mot_student/starter_code

PRED=$(ls -td ../outputs/seq_2_student_* | head -1)/result.csv

uv run --active python debug_tracking_viewer.py \
  --data-root ../test_input \
  --seq seq_2 \
  --result "$PRED" \
  --fps 8 \
  --tail 30
```

鍵盤控制：

| Key | 功能 |
|---|---|
| `Space` | 暫停 / 繼續 |
| `Left` / `Right` | 前一幀 / 下一幀 |
| `Home` / `End` | 跳到第一幀 / 最後一幀 |
| `Q` 或 `Esc` | 關閉 |

也可以匯出影片：

```bash
uv run --active python debug_tracking_viewer.py \
  --data-root ../test_input \
  --seq seq_2 \
  --result "$PRED" \
  --save-mp4 ../outputs/seq_2_debug.mp4 \
  --no-popup
```

MP4 匯出需要系統有 `ffmpeg`；若沒有，彈窗播放仍可使用。
