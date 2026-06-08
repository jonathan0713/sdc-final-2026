from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluate_tracking import load_tracking_csv


def parse_labels(text: str) -> np.ndarray:
    if text is None or str(text).strip() == "":
        return np.asarray([], dtype=np.float32)

    return np.asarray([float(v) for v in str(text).strip().split()], dtype=np.float32)


def load_tracking_csv_with_xyz(csv_path: str | Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    csv_path = Path(csv_path)
    masks: dict[int, np.ndarray] = {}
    xyz: dict[int, np.ndarray] = defaultdict(dict)

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "id" not in reader.fieldnames or "mask" not in reader.fieldnames:
            raise ValueError(f"CSV must contain columns: id, mask. File: {csv_path}")

        for row in reader:
            row_id = str(row["id"])

            if row_id.endswith("x") or row_id.endswith("y") or row_id.endswith("z"):
                frame_id = int(row_id[:-1])
                axis = row_id[-1]
                xyz[frame_id][axis] = parse_labels(row["mask"])
                continue

            frame_id = int(row_id)
            masks[frame_id] = parse_labels(row["mask"]).astype(np.int32)

    coords = {}
    for frame_id, axes in xyz.items():
        if {"x", "y", "z"}.issubset(axes):
            coords[frame_id] = np.column_stack([axes["x"], axes["y"], axes["z"]])

    return masks, coords


def infer_gt_path(data_root: str | Path, seq: str) -> Path:
    seq_number = seq.split("_")[-1]
    gt_path = Path(data_root) / seq / f"gt_answer_seq{seq_number}.csv"

    if not gt_path.exists():
        raise FileNotFoundError(f"Cannot infer GT path: {gt_path}. Pass --gt explicitly.")

    return gt_path


def find_latest_result(seq: str, output_root: str | Path) -> Path:
    output_root = Path(output_root)
    candidates = sorted(
        output_root.glob(f"{seq}_student_*/result.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No result.csv found under {output_root} for {seq}. "
            "Pass --pred explicitly."
        )

    return candidates[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def majority_label(labels: np.ndarray) -> tuple[int | None, int]:
    labels = labels[labels >= 0]

    if len(labels) == 0:
        return None, 0

    values, counts = np.unique(labels, return_counts=True)
    best_idx = int(np.argmax(counts))
    return int(values[best_idx]), int(counts[best_idx])


def summarize_ids(counter: Counter) -> str:
    return " ".join(f"{label}:{count}" for label, count in counter.most_common())


def gt_centroids_for_frame(gt_mask: np.ndarray, coords: np.ndarray | None) -> dict[int, np.ndarray]:
    if coords is None or len(coords) != len(gt_mask):
        return {}

    centroids = {}
    for gt_id in np.unique(gt_mask[gt_mask >= 0]):
        gt_id = int(gt_id)
        idx = gt_mask == gt_id
        if idx.sum() > 0:
            centroids[gt_id] = coords[idx, :2].mean(axis=0)

    return centroids


def main_pred_for_gt(gt_mask: np.ndarray, pred_mask: np.ndarray, gt_id: int) -> tuple[int | None, int, str]:
    idx = gt_mask == gt_id
    gt_points = int(idx.sum())

    if gt_points == 0:
        return None, 0, ""

    pred_counter = Counter(int(v) for v in pred_mask[idx] if int(v) >= 0)

    if not pred_counter:
        return None, 0, ""

    main_pred_id, matched_points = pred_counter.most_common(1)[0]
    return int(main_pred_id), int(matched_points), summarize_ids(pred_counter)


def build_id_switch_context(
    gt: dict[int, np.ndarray],
    pred: dict[int, np.ndarray],
    gt_coords: dict[int, np.ndarray],
    idsw_rows: list[dict],
) -> list[dict]:
    context_rows = []

    for event_idx, event in enumerate(idsw_rows, start=1):
        target_gt_id = int(event["gt_id"])

        for frame_id in range(int(event["start_frame"]), int(event["end_frame"]) + 1):
            if frame_id not in gt or frame_id not in pred:
                continue

            gt_mask = gt[frame_id]
            pred_mask = pred[frame_id]
            target_main_pred, target_matched_points, target_pred_ids = main_pred_for_gt(
                gt_mask,
                pred_mask,
                target_gt_id,
            )
            target_points = int((gt_mask == target_gt_id).sum())
            centroids = gt_centroids_for_frame(gt_mask, gt_coords.get(frame_id))

            nearest_gt_id = ""
            nearest_distance = ""
            nearest_main_pred = ""
            nearest_matched_points = 0
            nearest_pred_ids = ""

            if target_gt_id in centroids:
                nearest = []
                for other_gt_id, centroid in centroids.items():
                    if other_gt_id == target_gt_id:
                        continue
                    distance = float(np.linalg.norm(centroid - centroids[target_gt_id]))
                    nearest.append((distance, other_gt_id))

                if nearest:
                    distance, other_gt_id = min(nearest)
                    nearest_gt_id = other_gt_id
                    nearest_distance = round(distance, 4)
                    other_main_pred, other_matched_points, other_pred_ids = main_pred_for_gt(
                        gt_mask,
                        pred_mask,
                        other_gt_id,
                    )
                    nearest_main_pred = "" if other_main_pred is None else other_main_pred
                    nearest_matched_points = other_matched_points
                    nearest_pred_ids = other_pred_ids

            same_as_nearest = (
                target_main_pred is not None
                and nearest_main_pred != ""
                and int(nearest_main_pred) == int(target_main_pred)
            )

            context_rows.append(
                {
                    "event_index": event_idx,
                    "event_frame": event["frame"],
                    "frame": frame_id,
                    "relative_frame": frame_id - int(event["frame"]),
                    "gt_id": target_gt_id,
                    "event_prev_pred_id": event["prev_pred_id"],
                    "event_new_pred_id": event["new_pred_id"],
                    "target_main_pred_id": "" if target_main_pred is None else target_main_pred,
                    "target_gt_points": target_points,
                    "target_matched_points": target_matched_points,
                    "target_pred_ids": target_pred_ids,
                    "nearest_gt_id": nearest_gt_id,
                    "nearest_xy_distance": nearest_distance,
                    "nearest_main_pred_id": nearest_main_pred,
                    "nearest_matched_points": nearest_matched_points,
                    "nearest_pred_ids": nearest_pred_ids,
                    "same_main_pred_as_nearest": same_as_nearest,
                }
            )

    return context_rows


def analyze(
    gt_csv: str | Path,
    pred_csv: str | Path,
    window: int,
    min_overlap_points: int,
    close_distance: float,
) -> dict[str, list[dict]]:
    gt, gt_coords = load_tracking_csv_with_xyz(gt_csv)
    pred = load_tracking_csv(pred_csv)

    frame_rows = []
    idsw_rows = []
    fragmentation_rows = []
    merge_rows = []
    split_rows = []
    ghost_rows = []
    close_pair_rows = []
    gt_stats = defaultdict(
        lambda: {
            "frames_visible": 0,
            "points_total": 0,
            "missed_frames": 0,
            "missed_points": 0,
            "idsw_count": 0,
            "fragmentation_count": 0,
            "pred_counter": Counter(),
        }
    )
    pred_stats = defaultdict(
        lambda: {
            "frames": 0,
            "points_total": 0,
            "gt_counter": Counter(),
        }
    )

    prev_pred_for_gt: dict[int, int] = {}
    prev_gt_visible: dict[int, bool] = {}

    for frame_id in sorted(gt.keys()):
        if frame_id not in pred:
            raise ValueError(f"Missing frame {frame_id} in prediction CSV.")

        gt_mask = gt[frame_id]
        pred_mask = pred[frame_id]

        if len(gt_mask) != len(pred_mask):
            raise ValueError(
                f"Length mismatch at frame {frame_id}: gt={len(gt_mask)}, pred={len(pred_mask)}"
            )

        gt_pos = gt_mask >= 0
        pred_pos = pred_mask >= 0

        fp_points = int(((~gt_pos) & pred_pos).sum())
        fn_points = int((gt_pos & (~pred_pos)).sum())
        idsw_count = 0
        fragmentation_count = 0
        merge_count = 0
        split_count = 0
        ghost_count = 0

        current_gt_ids = set(int(v) for v in np.unique(gt_mask[gt_pos]))

        pred_to_gt: dict[int, Counter] = defaultdict(Counter)
        gt_to_pred: dict[int, Counter] = defaultdict(Counter)

        for gt_id, pred_id in zip(gt_mask, pred_mask):
            gt_id = int(gt_id)
            pred_id = int(pred_id)

            if gt_id >= 0 and pred_id >= 0:
                pred_to_gt[pred_id][gt_id] += 1
                gt_to_pred[gt_id][pred_id] += 1

        for pred_id, gt_counter in pred_to_gt.items():
            strong_gt_hits = Counter(
                {gt_id: count for gt_id, count in gt_counter.items() if count >= min_overlap_points}
            )

            if len(strong_gt_hits) < 2:
                continue

            merge_count += 1
            primary_gt_id, primary_points = strong_gt_hits.most_common(1)[0]
            merge_rows.append(
                {
                    "frame": frame_id,
                    "pred_id": pred_id,
                    "gt_ids": summarize_ids(strong_gt_hits),
                    "primary_gt_id": primary_gt_id,
                    "primary_points": primary_points,
                    "covered_gt_count": len(strong_gt_hits),
                    "overlap_points": sum(strong_gt_hits.values()),
                    "pred_points": int((pred_mask == pred_id).sum()),
                    "start_frame": max(0, frame_id - window),
                    "end_frame": frame_id + window,
                }
            )

        for gt_id, pred_counter in gt_to_pred.items():
            strong_pred_hits = Counter(
                {pred_id: count for pred_id, count in pred_counter.items() if count >= min_overlap_points}
            )

            if len(strong_pred_hits) < 2:
                continue

            split_count += 1
            primary_pred_id, primary_points = strong_pred_hits.most_common(1)[0]
            split_rows.append(
                {
                    "frame": frame_id,
                    "gt_id": gt_id,
                    "pred_ids": summarize_ids(strong_pred_hits),
                    "primary_pred_id": primary_pred_id,
                    "primary_points": primary_points,
                    "covered_pred_count": len(strong_pred_hits),
                    "overlap_points": sum(strong_pred_hits.values()),
                    "gt_points": int((gt_mask == gt_id).sum()),
                    "start_frame": max(0, frame_id - window),
                    "end_frame": frame_id + window,
                }
            )

        for pred_id in np.unique(pred_mask[pred_pos]):
            pred_id = int(pred_id)
            pred_idx = pred_mask == pred_id
            pred_points = int(pred_idx.sum())
            ghost_points = int((pred_idx & (~gt_pos)).sum())

            if ghost_points < min_overlap_points:
                continue

            ghost_count += 1
            ghost_rows.append(
                {
                    "frame": frame_id,
                    "pred_id": pred_id,
                    "ghost_points": ghost_points,
                    "pred_points": pred_points,
                    "ghost_ratio": round(ghost_points / max(pred_points, 1), 4),
                    "start_frame": max(0, frame_id - window),
                    "end_frame": frame_id + window,
                }
            )

        if frame_id in gt_coords and len(current_gt_ids) >= 2:
            coords = gt_coords[frame_id]
            centroids = {}

            for gt_id in current_gt_ids:
                idx = gt_mask == gt_id
                if idx.sum() == 0 or len(coords) != len(gt_mask):
                    continue
                centroids[gt_id] = coords[idx, :2].mean(axis=0)

            gt_ids = sorted(centroids)
            for idx_a, gt_id_a in enumerate(gt_ids):
                for gt_id_b in gt_ids[idx_a + 1 :]:
                    distance = float(np.linalg.norm(centroids[gt_id_a] - centroids[gt_id_b]))

                    if distance > close_distance:
                        continue

                    pred_a, points_a = majority_label(pred_mask[gt_mask == gt_id_a])
                    pred_b, points_b = majority_label(pred_mask[gt_mask == gt_id_b])
                    pred_ids_a = Counter(
                        int(v) for v in pred_mask[gt_mask == gt_id_a] if int(v) >= 0
                    )
                    pred_ids_b = Counter(
                        int(v) for v in pred_mask[gt_mask == gt_id_b] if int(v) >= 0
                    )

                    close_pair_rows.append(
                        {
                            "frame": frame_id,
                            "gt_id_a": gt_id_a,
                            "gt_id_b": gt_id_b,
                            "xy_distance": round(distance, 4),
                            "main_pred_a": "" if pred_a is None else pred_a,
                            "main_pred_b": "" if pred_b is None else pred_b,
                            "matched_points_a": points_a,
                            "matched_points_b": points_b,
                            "same_main_pred": bool(pred_a is not None and pred_a == pred_b),
                            "pred_ids_a": summarize_ids(pred_ids_a),
                            "pred_ids_b": summarize_ids(pred_ids_b),
                            "start_frame": max(0, frame_id - window),
                            "end_frame": frame_id + window,
                        }
                    )

        for pred_id in np.unique(pred_mask[pred_pos]):
            pred_id = int(pred_id)
            idx = pred_mask == pred_id
            pred_stats[pred_id]["frames"] += 1
            pred_stats[pred_id]["points_total"] += int(idx.sum())
            gt_labels = gt_mask[idx]
            gt_labels = gt_labels[gt_labels >= 0]

            if len(gt_labels) > 0:
                gt_values, gt_counts = np.unique(gt_labels, return_counts=True)
                for gt_id, count in zip(gt_values, gt_counts):
                    pred_stats[pred_id]["gt_counter"][int(gt_id)] += int(count)

        for gt_id in current_gt_ids:
            idx = gt_mask == gt_id
            point_count = int(idx.sum())
            gt_stats[gt_id]["frames_visible"] += 1
            gt_stats[gt_id]["points_total"] += point_count

            current_pred_id, overlap_points = majority_label(pred_mask[idx])

            if current_pred_id is None:
                gt_stats[gt_id]["missed_frames"] += 1
                gt_stats[gt_id]["missed_points"] += point_count
                prev_gt_visible[gt_id] = False
                continue

            gt_stats[gt_id]["pred_counter"][current_pred_id] += overlap_points

            if gt_id in prev_pred_for_gt and prev_pred_for_gt[gt_id] != current_pred_id:
                idsw_count += 1
                gt_stats[gt_id]["idsw_count"] += 1
                idsw_rows.append(
                    {
                        "frame": frame_id,
                        "gt_id": gt_id,
                        "prev_pred_id": prev_pred_for_gt[gt_id],
                        "new_pred_id": current_pred_id,
                        "gt_points": point_count,
                        "matched_points": overlap_points,
                        "start_frame": max(0, frame_id - window),
                        "end_frame": frame_id + window,
                    }
                )

            if gt_id in prev_gt_visible and prev_gt_visible[gt_id] is False:
                fragmentation_count += 1
                gt_stats[gt_id]["fragmentation_count"] += 1
                fragmentation_rows.append(
                    {
                        "frame": frame_id,
                        "gt_id": gt_id,
                        "pred_id": current_pred_id,
                        "gt_points": point_count,
                        "matched_points": overlap_points,
                        "start_frame": max(0, frame_id - window),
                        "end_frame": frame_id + window,
                    }
                )

            prev_pred_for_gt[gt_id] = current_pred_id
            prev_gt_visible[gt_id] = True

        for gt_id in list(prev_gt_visible.keys()):
            if gt_id not in current_gt_ids:
                prev_gt_visible[gt_id] = False

        frame_rows.append(
            {
                "frame": frame_id,
                "gt_positive_points": int(gt_pos.sum()),
                "pred_positive_points": int(pred_pos.sum()),
                "FP_points": fp_points,
                "FN_points": fn_points,
                "IDSW_events": idsw_count,
                "fragmentation_events": fragmentation_count,
                "merge_events": merge_count,
                "split_events": split_count,
                "ghost_tracks": ghost_count,
                "gt_objects": len(current_gt_ids),
                "pred_tracks": len(set(int(v) for v in np.unique(pred_mask[pred_pos]))),
            }
        )

    gt_rows = []
    for gt_id, stat in gt_stats.items():
        pred_counter = stat["pred_counter"]
        main_pred_id, main_pred_points = (None, 0)
        if pred_counter:
            main_pred_id, main_pred_points = pred_counter.most_common(1)[0]

        gt_rows.append(
            {
                "gt_id": gt_id,
                "frames_visible": stat["frames_visible"],
                "points_total": stat["points_total"],
                "missed_frames": stat["missed_frames"],
                "missed_points": stat["missed_points"],
                "IDSW_events": stat["idsw_count"],
                "fragmentation_events": stat["fragmentation_count"],
                "main_pred_id": "" if main_pred_id is None else main_pred_id,
                "main_pred_points": main_pred_points,
                "pred_ids_used": " ".join(str(k) for k, _ in pred_counter.most_common()),
            }
        )

    pred_rows = []
    for pred_id, stat in pred_stats.items():
        gt_counter = stat["gt_counter"]
        main_gt_id, main_gt_points = (None, 0)
        if gt_counter:
            main_gt_id, main_gt_points = gt_counter.most_common(1)[0]

        pred_rows.append(
            {
                "pred_id": pred_id,
                "frames": stat["frames"],
                "points_total": stat["points_total"],
                "main_gt_id": "" if main_gt_id is None else main_gt_id,
                "main_gt_points": main_gt_points,
                "gt_ids_covered": " ".join(str(k) for k, _ in gt_counter.most_common()),
            }
        )

    gt_rows.sort(key=lambda row: (-row["IDSW_events"], -row["fragmentation_events"], -row["points_total"]))
    pred_rows.sort(key=lambda row: (-row["points_total"], row["pred_id"]))
    idsw_rows.sort(key=lambda row: (row["frame"], row["gt_id"]))
    fragmentation_rows.sort(key=lambda row: (row["frame"], row["gt_id"]))
    merge_rows.sort(key=lambda row: (row["frame"], row["pred_id"]))
    split_rows.sort(key=lambda row: (row["frame"], row["gt_id"]))
    ghost_rows.sort(key=lambda row: (-row["ghost_points"], row["frame"], row["pred_id"]))
    close_pair_rows.sort(key=lambda row: (row["frame"], row["xy_distance"]))
    id_switch_context_rows = build_id_switch_context(gt, pred, gt_coords, idsw_rows)

    return {
        "frame_summary": frame_rows,
        "id_switch_events": idsw_rows,
        "id_switch_context": id_switch_context_rows,
        "fragmentation_events": fragmentation_rows,
        "merge_events": merge_rows,
        "split_events": split_rows,
        "ghost_track_events": ghost_rows,
        "close_pair_events": close_pair_rows,
        "gt_object_summary": gt_rows,
        "pred_track_summary": pred_rows,
    }


def event_comment(event: dict) -> str:
    if "prev_pred_id" in event:
        return (
            f"# id_switch frame={event['frame']} gt_id={event['gt_id']} "
            f"{event['prev_pred_id']}->{event['new_pred_id']}"
        )

    if "gt_ids" in event:
        return f"# merge frame={event['frame']} pred_id={event['pred_id']} gt_ids={event['gt_ids']}"

    if "pred_ids" in event:
        return f"# split frame={event['frame']} gt_id={event['gt_id']} pred_ids={event['pred_ids']}"

    if "gt_id_a" in event:
        return (
            f"# close_pair frame={event['frame']} gt={event['gt_id_a']},{event['gt_id_b']} "
            f"dist={event['xy_distance']} same_pred={event['same_main_pred']}"
        )

    if "ghost_points" in event:
        return (
            f"# ghost frame={event['frame']} pred_id={event['pred_id']} "
            f"ghost_points={event['ghost_points']}"
        )

    return f"# event frame={event.get('frame', '')}"


def write_viewer_commands(
    path: Path,
    data_root: str | Path | None,
    seq: str | None,
    pred_csv: str | Path,
    events: list[dict],
    limit: int,
) -> None:
    if data_root is None or seq is None:
        path.write_text("")
        return

    lines = []
    seen_windows = set()

    for event in events[:limit]:
        window_key = (event["start_frame"], event["end_frame"])
        if window_key in seen_windows:
            continue

        seen_windows.add(window_key)
        lines.append(event_comment(event))
        lines.append(
            "uv run --active python debug_tracking_viewer.py "
            f"--data-root {data_root} "
            f"--seq {seq} "
            f"--result {pred_csv} "
            f"--start {event['start_frame']} "
            f"--end {event['end_frame']} "
            "--fps 5 "
            "--tail 30"
        )
        lines.append("")

    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_all_viewer_commands(
    out_dir: Path,
    data_root: str | Path | None,
    seq: str | None,
    pred_csv: str | Path,
    tables: dict[str, list[dict]],
    limit: int,
) -> None:
    groups = {
        "viewer_commands.txt": (
            tables["id_switch_events"]
            + tables["merge_events"]
            + tables["split_events"]
            + tables["close_pair_events"]
            + tables["ghost_track_events"]
        ),
        "viewer_commands_id_switch.txt": tables["id_switch_events"],
        "viewer_commands_close_pairs.txt": tables["close_pair_events"],
        "viewer_commands_merge_split_ghost.txt": (
            tables["merge_events"] + tables["split_events"] + tables["ghost_track_events"]
        ),
    }

    for filename, events in groups.items():
        write_viewer_commands(
            path=out_dir / filename,
            data_root=data_root,
            seq=seq,
            pred_csv=pred_csv,
            events=events,
            limit=limit,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GT/pred tracking errors and produce debug tables.")
    parser.add_argument("--gt", default=None, help="GT CSV. Defaults to <data-root>/<seq>/gt_answer_seqN.csv.")
    parser.add_argument("--pred", default=None, help="Prediction CSV. Defaults to latest result under --output-root.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seq", default=None, choices=["seq_1", "seq_3"])
    parser.add_argument("--output-root", default="../outputs")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--viewer-limit", type=int, default=30)
    parser.add_argument("--min-overlap-points", type=int, default=2)
    parser.add_argument("--close-distance", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.gt is None:
        if args.data_root is None or args.seq is None:
            raise ValueError("Pass --gt or pass both --data-root and --seq.")
        gt_csv = infer_gt_path(args.data_root, args.seq)
    else:
        gt_csv = Path(args.gt)

    if args.pred is None:
        if args.seq is None:
            raise ValueError("Pass --pred or pass --seq so the latest result can be inferred.")
        pred_csv = find_latest_result(args.seq, args.output_root)
    else:
        pred_csv = Path(args.pred)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = analyze(
        gt_csv,
        pred_csv,
        args.window,
        min_overlap_points=args.min_overlap_points,
        close_distance=args.close_distance,
    )

    for name, rows in tables.items():
        write_csv(out_dir / f"{name}.csv", rows)

    write_all_viewer_commands(
        out_dir=out_dir,
        data_root=args.data_root,
        seq=args.seq,
        pred_csv=pred_csv,
        tables=tables,
        limit=args.viewer_limit,
    )

    print(f"gt: {gt_csv}")
    print(f"pred: {pred_csv}")
    print(f"out_dir: {out_dir}")
    print(f"frames: {len(tables['frame_summary'])}")
    print(f"id_switch_events: {len(tables['id_switch_events'])}")
    print(f"fragmentation_events: {len(tables['fragmentation_events'])}")
    print(f"merge_events: {len(tables['merge_events'])}")
    print(f"split_events: {len(tables['split_events'])}")
    print(f"ghost_track_events: {len(tables['ghost_track_events'])}")
    print(f"close_pair_events: {len(tables['close_pair_events'])}")
    print(f"[DONE] wrote analysis CSV files")
