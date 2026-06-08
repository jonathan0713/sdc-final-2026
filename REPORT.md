# SDC 2026 Spring Final Competition - Tracking 報告

## 1. 競賽目的

本次競賽主題是 **4D radar multi-object tracking, MOT**。參賽者需要根據連續時間序列中的 4D 雷達點雲與已給定的 moving cluster mask，為移動物體分配穩定且連續的 tracking ID。

競賽的核心不是從原始雷達點雲中偵測物體，而是針對已經被分群為移動群集的雷達點，完成跨幀資料關聯：

```text
4D radar point cloud
        ↓
moving cluster mask
        ↓
cluster-level detection
        ↓
multi-object tracking
        ↓
point-level tracking mask
        ↓
result.csv
```

如果同一個物體在多個 frame 中持續出現，演算法應該盡量維持相同的 track ID，避免 ID switch 與軌跡斷裂。

## 2. 競賽內容

本次共有兩個獨立競賽。

| Competition | 場景 | 開發序列 | Kaggle 測試序列 |
|---|---|---|---|
| Competition 1 | Static-ego tracking | `seq_1` | `seq_2` |
| Competition 2 | Moving-ego tracking | `seq_3` | `seq_4` |

### Competition 1: Static-Ego 4D Radar MOT

自車位置固定或近似固定，主要挑戰在於雷達點雲稀疏、cluster 形狀不穩定、不同物體接近時容易產生 ID switch。

資料位置：

```text
competition_1_static_mot_student/
├── public_dev/seq_1/
│   ├── gt_answer_seq1.csv
│   ├── image/
│   ├── mask_cluster/
│   └── radar/
└── test_input/seq_2/
    ├── image/
    ├── mask_cluster/
    └── radar/
```

### Competition 2: Moving-Ego 4D Radar MOT

自車會移動，因此物體在 ego coordinate 下的觀測位置會受到自車運動影響。此競賽額外提供 `ego_global_pos.txt`，可用於設計 ego-motion compensation。

資料位置：

```text
competition_2_moving_ego_mot_student/
├── public_dev/seq_3/
│   ├── ego_global_pos.txt
│   ├── gt_answer_seq3.csv
│   ├── image/
│   ├── mask_cluster/
│   └── radar/
└── test_input/seq_4/
    ├── ego_global_pos.txt
    ├── image/
    ├── mask_cluster/
    └── radar/
```

目前實作的 baseline 會讀取 ego motion 資訊，但 tracker 本身採用 radar cluster association，尚未明確做 ego-motion compensation。

## 3. 輸入資料細節

每個 frame 主要有兩份資料：

```text
radar/
mask_cluster/
```

### 3.1 Radar Point Cloud

每個 radar frame 是 binary file，每個點包含 5 個 float32 欄位：

| 欄位 | 說明 |
|---|---|
| `x` | 前方位置，forward position |
| `y` | 橫向位置，lateral position |
| `z` | 高度 |
| `RangeRate` | 徑向速度量測 |
| `RCS` | radar cross section |

資料格式：

```text
[x, y, z, RangeRate, RCS]
```

### 3.2 Cluster Mask

`mask_cluster/` 中每一列對應一個 radar point，表示該點屬於哪一個 cluster。

| Mask value | 意義 |
|---|---|
| `-1` | static point |
| `>= 0` | moving cluster ID |

競賽已經提供 moving cluster mask，因此演算法不需要自行做 object detection，只需要追蹤這些 moving cluster。

## 4. 輸出格式

Kaggle submission 需要輸出 `result.csv`，包含兩個欄位：

```text
id,mask
```

每個 frame 需要 4 rows：

```text
frame_id, tracking mask
frame_idx, x coordinates
frame_idy, y coordinates
frame_idz, z coordinates
```

tracking mask 的 label 定義：

| Output value | 意義 |
|---|---|
| `-1` | static point |
| `-2` | moving point, but not assigned to confirmed track |
| `>= 0` | confirmed track ID |

注意：如果真實 moving object 被標成 `-2`，會被視為 missed detection，增加 FN。

## 5. 實作流程

本專案已完成 starter code 中的主要 TODO，流程如下。

### 5.1 分離 static / moving points

使用 `sdc_tracking_utils.py` 中的 `separate_static_and_moving()`：

```text
mask == -1  -> static points
mask != -1  -> moving points
```

後續 tracking 只處理 moving points。

### 5.2 Cluster-level detection extraction

在 `main.py` 中完成 `calculate_cluster_centroids()`。

對每個 moving cluster：

1. 取出該 cluster 的所有 radar points。
2. 計算平均 `[x, y, z]` 作為 centroid。
3. 計算平均 `RangeRate` 作為速度特徵。
4. 保留原始 cluster points 供後續 mapping 或 visualization 使用。

產生的資料：

```python
cluster_centroids[cluster_id] = np.array([x, y, z])
cluster_velocities[cluster_id] = mean_range_rate
cluster_points_dict[cluster_id] = points
```

### 5.3 Detection array

在 `build_detections()` 中將 cluster dictionary 轉為 tracker 使用的 NumPy array：

```text
[x, y, z, cluster_id, range_rate]
```

shape 為：

```text
num_detections x 5
```

### 5.4 Tracker update

在 `sdc_tracker.py` 中完成 tracker lifecycle。

每個 track 保存：

| 欄位 | 說明 |
|---|---|
| `mean` | 目前預估的 `[x, y]` 位置 |
| `velocity` | 簡單 constant-velocity 預測 |
| `track_id` | 全域唯一 tracking ID |
| `hits` | 累積成功匹配次數 |
| `age` | 未匹配幀數 |
| `state` | `tentative`, `confirmed`, `deleted` |
| `det` | 最近一次匹配的 detection |
| `updated` | 此 track 在當前 frame 是否有成功匹配 |
| `prev_states` | trajectory history |

### 5.5 Data association

在 `match_tracks()` 中使用 Hungarian assignment 做全域配對。

流程：

1. 對每個 track 做簡單位置預測：

```text
predicted_xy = mean + velocity * age
```

2. 計算 track 與 detection 的 cost：

```text
cost = xy_distance + 0.1 * z_distance + 0.05 * range_rate_distance
```

3. 使用 `max_distance` 做 gating，超出距離的配對不允許。
4. 使用 `scipy.optimize.linear_sum_assignment()` 找最低總成本配對。
5. 未匹配 detection 建立新 track。
6. 未匹配 track 增加 age，超過 `max_age` 後刪除。

### 5.6 Track to cluster mapping

tracker 回傳 confirmed tracks：

```text
[x, y, z, cluster_id, track_id]
```

`generate_cluster_track_dict()` 會轉成：

```python
{
    cluster_id: track_id
}
```

最後由 `save_tracking_mask_to_csv()` 將 cluster-level track ID 回填到每個 radar point，產生符合 Kaggle 格式的 `result.csv`。

## 6. 預設參數

目前預設 tracker 參數：

```text
max_age = 5
min_hits = 1
max_distance = 5.0
```

說明：

| 參數 | 效果 |
|---|---|
| `max_age` | track 最多可失配幾幀仍保留 |
| `min_hits` | track 至少匹配幾次才 confirmed |
| `max_distance` | association 最大允許距離 |

目前使用 `min_hits=1` 是因為 competition 的 moving cluster mask 已經提供，如果延遲 confirmed，許多真實 moving points 會被輸出成 `-2`，造成 FN 增加。

## 7. 評估方式

只有 public development sequences 有 ground truth：

| Competition | Dev sequence | Ground truth |
|---|---|---|
| Competition 1 | `seq_1` | `gt_answer_seq1.csv` |
| Competition 2 | `seq_3` | `gt_answer_seq3.csv` |

Kaggle test sequences `seq_2` 與 `seq_4` 沒有 ground truth，只能做格式檢查與人工可視化檢查。

### 7.1 評估指標

`evaluate_tracking.py` 會輸出：

| 指標 | 說明 |
|---|---|
| `gt_positive_points` | ground truth 中 moving object points 數量 |
| `FP_points` | 預測為 moving track，但 GT 不是 moving 的點 |
| `FN_points` | GT 是 moving，但預測沒有 confirmed track 的點 |
| `IDSW_proxy` | ID switch proxy |
| `fragmentation_proxy` | 軌跡中斷 proxy |
| `MOTA_style_score` | 類 MOTA 分數，越高越好 |

### 7.2 目前本機驗證結果

| Competition | Dev Seq | Format | MOTA-style score | FP | FN | IDSW proxy | Fragmentation proxy |
|---|---:|---|---:|---:|---:|---:|---:|
| Static Ego | `seq_1` | OK | 0.984788 | 0 | 0 | 153 | 409 |
| Moving Ego | `seq_3` | OK | 0.989988 | 0 | 0 | 97 | 356 |

### 7.3 Test submission 格式檢查

| Competition | Test Seq | Rows | Result |
|---|---:|---:|---|
| Static Ego | `seq_2` | 1284 | OK |
| Moving Ego | `seq_4` | 2400 | OK |

目前已產生的提交檔：

```text
competition_1_static_mot_student/outputs/seq_2_student_20260602_163108/result.csv
competition_2_moving_ego_mot_student/outputs/seq_4_student_20260602_162724/result.csv
```

## 8. 使用 uv 執行

建議在專案根目錄建立獨立 Python 環境：

```bash
cd /home/jonathan/Documents/ROS1/sdc-final-2026

uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install -r competition_1_static_mot_student/starter_code/requirements.txt
```

### 8.1 Competition 1: 產生 seq_2 submission

```bash
cd /home/jonathan/Documents/ROS1/sdc-final-2026/competition_1_static_mot_student/starter_code

uv run --active python main.py \
  --data-root ../test_input \
  --seq seq_2 \
  --output-root ../outputs

PRED=$(ls -td ../outputs/seq_2_student_* | head -1)/result.csv

uv run --active python check_result_format.py \
  --submission "$PRED" \
  --data-root ../test_input \
  --seq seq_2
```

### 8.2 Competition 2: 產生 seq_4 submission

```bash
cd /home/jonathan/Documents/ROS1/sdc-final-2026/competition_2_moving_ego_mot_student/starter_code

uv run --active python main.py \
  --data-root ../test_input \
  --seq seq_4 \
  --output-root ../outputs

PRED=$(ls -td ../outputs/seq_4_student_* | head -1)/result.csv

uv run --active python check_result_format.py \
  --submission "$PRED" \
  --data-root ../test_input \
  --seq seq_4
```

## 9. 可視化與 Debug

原本使用 `--plot-every` 產生一批靜態 PNG，雖然能看到單幀結果，但不適合人工 debug tracking，因為 ID switch、軌跡斷裂、短暫失配通常要看連續時間變化。因此目前新增 `debug_tracking_viewer.py`，以彈窗影片方式同步顯示 camera image 與 radar bird's-eye tracking。

### 9.1 Interactive camera + radar playback

範例：播放 Competition 1 的 `seq_2` 最新 tracking 結果。

```bash
cd /home/jonathan/Documents/ROS1/sdc-final-2026/competition_1_static_mot_student/starter_code

PRED=$(ls -td ../outputs/seq_2_student_* | head -1)/result.csv

uv run --active python debug_tracking_viewer.py \
  --data-root ../test_input \
  --seq seq_2 \
  --result "$PRED" \
  --fps 8 \
  --tail 30
```

彈窗左側顯示對應 frame 的 camera image，右側顯示 radar bird's-eye tracking。鍵盤控制如下：

| Key | 功能 |
|---|---|
| `Space` | 暫停 / 繼續播放 |
| `Left` / `Right` | 前一幀 / 下一幀 |
| `Home` / `End` | 跳到第一幀 / 最後一幀 |
| `Q` 或 `Esc` | 關閉視窗 |

圖中包含：

| 可視化項目 | 說明 |
|---|---|
| 左側相機圖 | 對應 frame 的 camera image |
| 灰色點 | static radar points |
| 橘色點 | moving points, but unconfirmed (`-2`) |
| 彩色點 | confirmed track radar points |
| 彩色 ID 標籤 | confirmed track ID |
| 軌跡線 | 最近 `--tail` 幀的 track centroid history |
| X/Y 座標 | bird's-eye view 中的 forward/lateral 位置 |

這可以用來人工檢查：

1. 同一個物體的 ID 是否保持一致。
2. 物體靠近時是否發生 ID switch。
3. 新物體出現時是否快速建立 track。
4. 物體短暫消失後是否能接回原 track。
5. cluster mask 是否出現破碎或跳動。
6. camera image 中是否真的存在對應移動物體。

### 9.2 匯出 MP4

若需要交報告或分享 debug 結果，可以直接匯出影片：

```bash
uv run --active python debug_tracking_viewer.py \
  --data-root ../test_input \
  --seq seq_2 \
  --result "$PRED" \
  --save-mp4 ../outputs/seq_2_debug.mp4 \
  --no-popup
```

MP4 匯出使用 Matplotlib 的 `FFMpegWriter`，系統需要安裝 `ffmpeg`。如果沒有 `ffmpeg`，仍可使用彈窗播放，不影響 `result.csv` 產生與 Kaggle submission。

### 9.3 Competition 2 playback

Moving-ego competition 的使用方式相同，只需換資料根目錄與 sequence：

```bash
cd /home/jonathan/Documents/ROS1/sdc-final-2026/competition_2_moving_ego_mot_student/starter_code

PRED=$(ls -td ../outputs/seq_4_student_* | head -1)/result.csv

uv run --active python debug_tracking_viewer.py \
  --data-root ../test_input \
  --seq seq_4 \
  --result "$PRED" \
  --fps 8 \
  --tail 30
```

### 9.4 評估 CSV

public dev sequence 可產生評估 CSV：

```bash
uv run --active python evaluate_tracking.py \
  --gt ../public_dev/seq_1/gt_answer_seq1.csv \
  --pred "$PRED" \
  --out ../outputs/seq_1_eval.csv
```

可視化或分析時，可優先查看：

```text
../outputs/seq_1_eval.csv
../outputs/seq_3_eval.csv
```

### 9.5 Submission format check

對 hidden test 沒有 ground truth，因此至少要檢查 submission 格式：

```bash
uv run --active python check_result_format.py \
  --submission "$PRED" \
  --data-root ../test_input \
  --seq seq_2
```

成功時會看到：

```text
[OK] submission format is valid
```

### 9.6 PNG fallback

`main.py --plot-every` 仍保留作為備用，可每隔固定 frame 輸出 radar-only PNG。不過建議主要使用 `debug_tracking_viewer.py`，因為它能看連續播放，也能同時對照 camera image。

### 9.7 可進一步增加的視覺化

若要更深入分析，可以擴充以下工具：

| 可視化 | 用途 |
|---|---|
| GT vs prediction overlay | 在 public dev 上直接比較 GT ID 與 predicted ID |
| ID switch timeline | 找出 ID switch 發生在哪些 frame |
| track length histogram | 查看 track 是否太碎 |
| per-frame FP/FN plot | 找出特定 frame 是否有大量錯誤 |
| ego path plot | Competition 2 中檢查自車運動 |
| cluster centroid trajectory | 看 detection centroid 是否跳動 |

## 10. 目前方法的優點與限制

### 優點

1. 完成 starter code 所有主要 TODO，可直接產生 Kaggle submission。
2. cluster-level tracking 簡潔穩定，public dev 上 FP/FN 為 0。
3. 使用 Hungarian assignment，比 greedy nearest neighbor 更能處理多物體同時出現。
4. 支援互動式 camera + radar playback，方便人工檢查 ID。
5. 不依賴 pandas，降低 Python 環境相容問題。

### 限制

1. Competition 2 尚未明確使用 ego-motion compensation。
2. motion model 是簡單 constant velocity，沒有 Kalman filter covariance。
3. association cost 只使用 centroid、z、range rate，尚未使用 cluster shape、RCS 或點數等特徵。
4. camera image 尚未與 radar tracking 結果融合。
5. hidden test 的真實分數無法本機得知，只能透過 Kaggle 評估。

## 11. 改進方向

後續若要提升分數，可優先嘗試：

1. 對 Competition 2 加入 ego-motion compensation。
2. 使用 Kalman filter 取代簡單 constant velocity。
3. 在 association cost 加入 cluster size、RCS mean、cluster covariance。
4. 對 `max_distance`, `max_age`, `min_hits` 做 grid search。
5. 在 public dev 上建立 ID switch debug tool，自動列出錯誤 frame。
6. 增加 GT/pred overlay visualization，快速定位 ID switch 原因。

## 12. 結論

本專案已完成一個可提交的 4D radar MOT baseline。方法將 moving radar clusters 轉為 cluster-level detections，透過 Hungarian assignment 與簡單 motion prediction 維持跨幀 track ID，最後輸出 point-level `result.csv`。

目前在 public dev 上的 MOTA-style score 分別為：

```text
seq_1: 0.984788
seq_3: 0.989988
```

兩個 Kaggle test sequences 的 submission format 皆已通過檢查。可提交的檔案為：

```text
competition_1_static_mot_student/outputs/seq_2_student_20260602_163108/result.csv
competition_2_moving_ego_mot_student/outputs/seq_4_student_20260602_162724/result.csv
```
