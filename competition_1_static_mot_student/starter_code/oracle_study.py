from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_errors import analyze, infer_gt_path
from evaluate_tracking import load_tracking_csv


def find_latest_result(seq: str, output_root: str | Path) -> Path:
    output_root = Path(output_root)
    candidates = sorted(
        output_root.glob(f"{seq}_student_*/result.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No result.csv found under {output_root} for {seq}. Pass --pred explicitly.")

    return candidates[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("")
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_masks(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray]) -> dict:
    fp_points = 0
    fn_points = 0
    idsw_proxy = 0
    fragmentation_proxy = 0
    gt_positive_points = 0

    prev_pred_for_gt: dict[int, int] = {}
    prev_gt_visible: dict[int, bool] = {}

    for frame_id in sorted(gt):
        if frame_id not in pred:
            raise ValueError(f"Missing frame {frame_id} in prediction masks.")

        gt_mask = gt[frame_id]
        pred_mask = pred[frame_id]

        if len(gt_mask) != len(pred_mask):
            raise ValueError(f"Length mismatch at frame {frame_id}: gt={len(gt_mask)}, pred={len(pred_mask)}")

        gt_pos = gt_mask >= 0
        pred_pos = pred_mask >= 0
        gt_positive_points += int(gt_pos.sum())
        fp_points += int(((~gt_pos) & pred_pos).sum())
        fn_points += int((gt_pos & (~pred_pos)).sum())

        current_gt_ids = set(int(v) for v in np.unique(gt_mask[gt_pos]))

        for gt_id in current_gt_ids:
            idx = gt_mask == gt_id
            pred_ids = pred_mask[idx]
            pred_ids = pred_ids[pred_ids >= 0]

            if len(pred_ids) == 0:
                prev_gt_visible[gt_id] = False
                continue

            values, counts = np.unique(pred_ids, return_counts=True)
            current_pred_id = int(values[np.argmax(counts)])

            if gt_id in prev_pred_for_gt and prev_pred_for_gt[gt_id] != current_pred_id:
                idsw_proxy += 1

            if gt_id in prev_gt_visible and prev_gt_visible[gt_id] is False:
                fragmentation_proxy += 1

            prev_pred_for_gt[gt_id] = current_pred_id
            prev_gt_visible[gt_id] = True

        for gt_id in list(prev_gt_visible):
            if gt_id not in current_gt_ids:
                prev_gt_visible[gt_id] = False

    score = 1.0 - (fp_points + fn_points + idsw_proxy) / max(gt_positive_points, 1)

    return {
        "frames": len(gt),
        "gt_positive_points": gt_positive_points,
        "FP_points": fp_points,
        "FN_points": fn_points,
        "IDSW_proxy": idsw_proxy,
        "fragmentation_proxy": fragmentation_proxy,
        "MOTA_style_score": float(score),
    }


def build_global_majority_map(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray]) -> dict[int, int]:
    overlaps = defaultdict(Counter)

    for frame_id in sorted(gt):
        gt_mask = gt[frame_id]
        pred_mask = pred[frame_id]
        valid = (gt_mask >= 0) & (pred_mask >= 0)

        for gt_id, pred_id in zip(gt_mask[valid], pred_mask[valid]):
            overlaps[int(pred_id)][int(gt_id)] += 1

    mapping = {}
    for pred_id, counter in overlaps.items():
        if counter:
            mapping[pred_id] = counter.most_common(1)[0][0]

    return mapping


def make_global_relabel_oracle(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    mapping = build_global_majority_map(gt, pred)
    relabeled = {}

    for frame_id, pred_mask in pred.items():
        out = pred_mask.copy()
        for pred_id, gt_id in mapping.items():
            out[pred_mask == pred_id] = gt_id
        relabeled[frame_id] = out

    return relabeled


def make_frame_cluster_oracle(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    relabeled = {}

    for frame_id in sorted(gt):
        gt_mask = gt[frame_id]
        pred_mask = pred[frame_id]
        out = pred_mask.copy()

        for pred_id in np.unique(pred_mask[pred_mask >= 0]):
            pred_id = int(pred_id)
            idx = pred_mask == pred_id
            gt_ids = gt_mask[idx]
            gt_ids = gt_ids[gt_ids >= 0]

            if len(gt_ids) == 0:
                continue

            values, counts = np.unique(gt_ids, return_counts=True)
            out[idx] = int(values[np.argmax(counts)])

        relabeled[frame_id] = out

    return relabeled


def make_detected_point_identity_oracle(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    relabeled = {}

    for frame_id in sorted(gt):
        gt_mask = gt[frame_id]
        pred_mask = pred[frame_id]
        out = pred_mask.copy()
        detected_gt = (gt_mask >= 0) & (pred_mask >= 0)
        out[detected_gt] = gt_mask[detected_gt]
        relabeled[frame_id] = out

    return relabeled


def make_no_fp_oracle(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    relabeled = make_detected_point_identity_oracle(gt, pred)

    for frame_id in sorted(gt):
        out = relabeled[frame_id].copy()
        out[gt[frame_id] < 0] = -1
        relabeled[frame_id] = out

    return relabeled


def make_no_fn_oracle(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    relabeled = make_detected_point_identity_oracle(gt, pred)

    for frame_id in sorted(gt):
        out = relabeled[frame_id].copy()
        gt_pos = gt[frame_id] >= 0
        out[gt_pos] = gt[frame_id][gt_pos]
        relabeled[frame_id] = out

    return relabeled


def make_perfect_oracle(gt: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {frame_id: gt_mask.copy() for frame_id, gt_mask in gt.items()}


def value_to_float(value, default: float | None = None) -> float | None:
    if value == "" or value is None:
        return default

    return float(value)


def value_to_int(value, default: int = 0) -> int:
    if value == "" or value is None:
        return default

    return int(value)


def summarize_id_switch_categories(
    idsw_rows: list[dict],
    context_rows: list[dict],
    low_point_threshold: int,
    close_distance: float,
) -> tuple[list[dict], list[dict]]:
    contexts_by_event = defaultdict(list)

    for row in context_rows:
        contexts_by_event[int(row["event_index"])].append(row)

    enriched_rows = []
    category_counts = Counter()

    for event_index, event in enumerate(idsw_rows, start=1):
        contexts = contexts_by_event[event_index]
        event_context = next((row for row in contexts if int(row["relative_frame"]) == 0), None)
        prev_context = next((row for row in contexts if int(row["relative_frame"]) == -1), None)

        gt_points = int(event["gt_points"])
        matched_points = int(event["matched_points"])
        nearest_distance = value_to_float(event_context.get("nearest_xy_distance") if event_context else None)
        nearest_gt_id = "" if event_context is None else event_context.get("nearest_gt_id", "")
        nearest_main_pred_id = "" if event_context is None else event_context.get("nearest_main_pred_id", "")
        visible_contexts = [row for row in contexts if value_to_int(row["target_gt_points"]) > 0]
        detected_contexts = [row for row in contexts if value_to_int(row["target_matched_points"]) > 0]
        point_counts = [value_to_int(row["target_gt_points"]) for row in visible_contexts]
        pred_ids = Counter(
            str(row["target_main_pred_id"])
            for row in detected_contexts
            if str(row["target_main_pred_id"]) != ""
        )

        low_point_event = gt_points <= low_point_threshold
        close_pair_event = nearest_distance is not None and nearest_distance <= close_distance
        previous_missing_gt = prev_context is None or value_to_int(prev_context["target_gt_points"]) == 0
        previous_missed_detection = (
            prev_context is not None
            and value_to_int(prev_context["target_gt_points"]) > 0
            and value_to_int(prev_context["target_matched_points"]) == 0
        )

        if low_point_event:
            category_counts["low_point_event"] += 1
        if close_pair_event:
            category_counts["close_pair_event"] += 1
        if previous_missing_gt:
            category_counts["after_missing_gt"] += 1
        if previous_missed_detection:
            category_counts["after_missed_detection"] += 1
        if not (low_point_event or close_pair_event or previous_missing_gt or previous_missed_detection):
            category_counts["other"] += 1

        enriched_rows.append(
            {
                "event_index": event_index,
                "frame": event["frame"],
                "gt_id": event["gt_id"],
                "prev_pred_id": event["prev_pred_id"],
                "new_pred_id": event["new_pred_id"],
                "gt_points": gt_points,
                "matched_points": matched_points,
                "low_point_event": low_point_event,
                "nearest_gt_id": nearest_gt_id,
                "nearest_xy_distance": "" if nearest_distance is None else round(nearest_distance, 4),
                "nearest_main_pred_id": nearest_main_pred_id,
                "close_pair_event": close_pair_event,
                "previous_missing_gt": previous_missing_gt,
                "previous_missed_detection": previous_missed_detection,
                "visible_frames_in_window": len(visible_contexts),
                "detected_frames_in_window": len(detected_contexts),
                "min_gt_points_in_window": min(point_counts) if point_counts else 0,
                "median_gt_points_in_window": float(np.median(point_counts)) if point_counts else 0.0,
                "max_gt_points_in_window": max(point_counts) if point_counts else 0,
                "pred_ids_in_window": " ".join(f"{pred_id}:{count}" for pred_id, count in pred_ids.most_common()),
            }
        )

    total = max(len(idsw_rows), 1)
    summary_rows = []
    for category in [
        "low_point_event",
        "close_pair_event",
        "after_missing_gt",
        "after_missed_detection",
        "other",
    ]:
        count = category_counts[category]
        summary_rows.append(
            {
                "category": category,
                "IDSW_events": count,
                "ratio": round(count / total, 4),
            }
        )

    return summary_rows, enriched_rows


def metric_rows(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray]) -> list[dict]:
    scenarios = [
        (
            "baseline_prediction",
            "Current result.csv exactly as submitted.",
            pred,
        ),
        (
            "global_track_majority_oracle",
            "Relabel each predicted track id to the GT id it overlaps most over the whole sequence.",
            make_global_relabel_oracle(gt, pred),
        ),
        (
            "frame_cluster_majority_oracle",
            "Relabel each per-frame predicted cluster to its majority GT id. Cannot split merged clusters.",
            make_frame_cluster_oracle(gt, pred),
        ),
        (
            "detected_point_identity_oracle",
            "For GT points that were detected, replace predicted id with GT id. FP/FN coverage stays unchanged.",
            make_detected_point_identity_oracle(gt, pred),
        ),
        (
            "detected_point_identity_no_fp_oracle",
            "Same as detected-point identity, but remove false-positive background points.",
            make_no_fp_oracle(gt, pred),
        ),
        (
            "gt_complete_with_baseline_fp_oracle",
            "Fill all missed GT points with GT ids while keeping baseline FP points.",
            make_no_fn_oracle(gt, pred),
        ),
        (
            "perfect_gt_oracle",
            "Use GT labels directly. This is the absolute scoring ceiling.",
            make_perfect_oracle(gt),
        ),
    ]

    rows = []
    baseline_score = None

    for name, description, masks in scenarios:
        metrics = evaluate_masks(gt, masks)

        if baseline_score is None:
            baseline_score = metrics["MOTA_style_score"]

        rows.append(
            {
                "scenario": name,
                "score_gain_vs_baseline": round(metrics["MOTA_style_score"] - baseline_score, 8),
                **metrics,
                "description": description,
            }
        )

    return rows


def write_report(path: Path, metrics: list[dict], category_rows: list[dict], pred_csv: Path, gt_csv: Path) -> None:
    baseline = metrics[0]
    by_name = {row["scenario"]: row for row in metrics}
    identity = by_name["detected_point_identity_oracle"]
    cluster = by_name["frame_cluster_majority_oracle"]
    no_fp = by_name["detected_point_identity_no_fp_oracle"]
    no_fn = by_name["gt_complete_with_baseline_fp_oracle"]

    category_text = "\n".join(
        f"- {row['category']}: {row['IDSW_events']} ({row['ratio']:.1%})" for row in category_rows
    )

    lines = [
        "Oracle Study",
        "============",
        "",
        f"GT: {gt_csv}",
        f"Prediction: {pred_csv}",
        "",
        "Score ceilings",
        "--------------",
        f"- baseline score: {baseline['MOTA_style_score']:.8f}",
        f"- frame-cluster association oracle: {cluster['MOTA_style_score']:.8f} "
        f"(gain {cluster['score_gain_vs_baseline']:.8f})",
        f"- detected-point ID oracle: {identity['MOTA_style_score']:.8f} "
        f"(gain {identity['score_gain_vs_baseline']:.8f})",
        f"- detected-point ID + no FP oracle: {no_fp['MOTA_style_score']:.8f} "
        f"(gain {no_fp['score_gain_vs_baseline']:.8f})",
        f"- complete GT + baseline FP oracle: {no_fn['MOTA_style_score']:.8f} "
        f"(gain {no_fn['score_gain_vs_baseline']:.8f})",
        "",
        "ID switch categories",
        "--------------------",
        category_text,
        "",
        "Interpretation",
        "--------------",
    ]

    id_gain = identity["score_gain_vs_baseline"]
    cluster_gain = cluster["score_gain_vs_baseline"]
    fp_gap = no_fp["score_gain_vs_baseline"] - identity["score_gain_vs_baseline"]
    fn_gap = no_fn["score_gain_vs_baseline"] - identity["score_gain_vs_baseline"]

    if id_gain > 0.0001:
        lines.append("- ID continuity still has measurable room; prioritize re-ID/association rules.")
    else:
        lines.append("- Pure ID continuity has little measured room on this dev prediction.")

    if cluster_gain < id_gain * 0.5:
        lines.append("- A large part of the theoretical ID gain needs point/cluster splitting, so it may be hard to realize in the current cluster-level output.")
    else:
        lines.append("- Frame-level cluster relabeling captures much of the ID gain; association logic is a plausible next target.")

    if fp_gap > fn_gap and fp_gap > 0.0001:
        lines.append("- False positives are a larger remaining coverage term than missed GT points.")
    elif fn_gap > 0.0001:
        lines.append("- Missed GT points are a larger remaining coverage term than false positives.")
    else:
        lines.append("- FP/FN coverage gaps are small compared with the current score.")

    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run GT-only oracle studies on a dev tracking result.")
    parser.add_argument("--gt", default=None, help="GT CSV. Defaults to <data-root>/<seq>/gt_answer_seqN.csv.")
    parser.add_argument("--pred", default=None, help="Prediction CSV. Defaults to latest result under --output-root.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seq", default=None, choices=["seq_1", "seq_3"])
    parser.add_argument("--output-root", default="../outputs")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--low-point-threshold", type=int, default=3)
    parser.add_argument("--close-distance", type=float, default=2.0)
    parser.add_argument("--min-overlap-points", type=int, default=2)
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

    gt = load_tracking_csv(gt_csv)
    pred = load_tracking_csv(pred_csv)

    metrics = metric_rows(gt, pred)
    error_tables = analyze(
        gt_csv,
        pred_csv,
        window=args.window,
        min_overlap_points=args.min_overlap_points,
        close_distance=args.close_distance,
    )
    category_rows, enriched_rows = summarize_id_switch_categories(
        error_tables["id_switch_events"],
        error_tables["id_switch_context"],
        low_point_threshold=args.low_point_threshold,
        close_distance=args.close_distance,
    )

    write_csv(out_dir / "oracle_metrics.csv", metrics)
    write_csv(out_dir / "id_switch_category_summary.csv", category_rows)
    write_csv(out_dir / "id_switch_events_enriched.csv", enriched_rows)
    write_report(out_dir / "oracle_report.txt", metrics, category_rows, pred_csv, gt_csv)

    print(f"gt: {gt_csv}")
    print(f"pred: {pred_csv}")
    print(f"out_dir: {out_dir}")
    print(f"baseline_score: {metrics[0]['MOTA_style_score']:.8f}")
    print(
        "detected_point_identity_oracle: "
        f"{metrics[3]['MOTA_style_score']:.8f} "
        f"(gain {metrics[3]['score_gain_vs_baseline']:.8f})"
    )
    print(f"id_switch_events: {len(error_tables['id_switch_events'])}")
    print("[DONE] wrote oracle study files")
