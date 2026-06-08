from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_values(text: str, dtype=float) -> np.ndarray:
    if text is None or str(text).strip() == "":
        return np.asarray([], dtype=dtype)

    return np.asarray([dtype(float(value)) for value in str(text).strip().split()])


def load_result_csv(csv_path: str | Path) -> tuple[dict[int, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    csv_path = Path(csv_path)
    masks = {}
    coords = defaultdict(dict)

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "id" not in reader.fieldnames or "mask" not in reader.fieldnames:
            raise ValueError(f"CSV must contain id,mask columns: {csv_path}")

        for row in reader:
            row_id = str(row["id"])

            if row_id.endswith("x") or row_id.endswith("y") or row_id.endswith("z"):
                coords[int(row_id[:-1])][row_id[-1]] = parse_values(row["mask"], float)
            else:
                masks[int(row_id)] = parse_values(row["mask"], int).astype(np.int32)

    return masks, coords


def write_result_csv(path: str | Path, masks: dict[int, np.ndarray], coords: dict[int, dict[str, np.ndarray]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "mask"])
        writer.writeheader()

        for frame_id in sorted(masks):
            writer.writerow({"id": frame_id, "mask": " ".join(map(str, masks[frame_id].astype(int)))})

            for axis in ["x", "y", "z"]:
                if axis in coords.get(frame_id, {}):
                    values = coords[frame_id][axis]
                    writer.writerow({"id": f"{frame_id}{axis}", "mask": " ".join(map(str, values))})


def frame_xy(coords: dict[int, dict[str, np.ndarray]], frame_id: int) -> np.ndarray | None:
    axes = coords.get(frame_id, {})

    if "x" not in axes or "y" not in axes:
        return None

    return np.column_stack([axes["x"], axes["y"]])


def split_observations(
    track_id: int,
    observations: list[dict],
    split_gap: int,
    split_jump_distance: float,
) -> list[list[dict]]:
    if not observations:
        return []

    segments = [[observations[0]]]

    for obs in observations[1:]:
        prev = segments[-1][-1]
        frame_gap = int(obs["frame"]) - int(prev["frame"])
        jump_distance = float(np.linalg.norm(obs["xy"] - prev["xy"]))
        should_split = False

        if split_gap > 0 and frame_gap > split_gap:
            should_split = True

        if split_jump_distance > 0 and jump_distance > split_jump_distance:
            should_split = True

        if should_split:
            segments.append([obs])
        else:
            segments[-1].append(obs)

    return segments


def build_tracklets(
    masks: dict[int, np.ndarray],
    coords: dict[int, dict[str, np.ndarray]],
    split_gap: int = 0,
    split_jump_distance: float = 0.0,
) -> dict[int, dict]:
    observations = defaultdict(list)

    for frame_id in sorted(masks):
        mask = masks[frame_id]
        xy = frame_xy(coords, frame_id)

        if xy is None or len(xy) != len(mask):
            continue

        for track_id in np.unique(mask[mask >= 0]):
            track_id = int(track_id)
            idx = mask == track_id
            observations[track_id].append(
                {
                    "frame": frame_id,
                    "count": int(idx.sum()),
                    "xy": xy[idx].mean(axis=0),
                }
            )

    tracklets = {}
    segment_id = 0

    for track_id, obs in observations.items():
        obs = sorted(obs, key=lambda row: row["frame"])
        segments = split_observations(track_id, obs, split_gap, split_jump_distance)

        for segment_index, segment_obs in enumerate(segments):
            frames = [row["frame"] for row in segment_obs]
            tracklets[segment_id] = {
                "segment_id": segment_id,
                "track_id": track_id,
                "segment_index": segment_index,
                "observations": segment_obs,
                "start": frames[0],
                "end": frames[-1],
                "start_xy": segment_obs[0]["xy"],
                "end_xy": segment_obs[-1]["xy"],
                "start_count": segment_obs[0]["count"],
                "end_count": segment_obs[-1]["count"],
                "frames": set(frames),
                "length": len(frames),
                "velocity": estimate_velocity(segment_obs),
            }
            segment_id += 1

    return tracklets


def estimate_velocity(observations: list[dict], tail: int = 4) -> np.ndarray:
    if len(observations) < 2:
        return np.zeros(2, dtype=float)

    obs = observations[-tail:]
    first = obs[0]
    last = obs[-1]
    dt = max(int(last["frame"]) - int(first["frame"]), 1)
    return (last["xy"] - first["xy"]) / dt


def has_frame_overlap(track_a: dict, track_b: dict) -> bool:
    return bool(track_a["frames"] & track_b["frames"])


def find_merge_pairs(
    tracklets: dict[int, dict],
    max_gap: int,
    max_distance: float,
    low_point_threshold: int,
    min_source_length: int,
    max_cost: float,
    max_target_length: int,
    max_velocity_diff: float,
    velocity_weight: float,
    cross_id_only: bool,
) -> list[dict]:
    pairs = []
    active_tracklets = sorted(tracklets.values(), key=lambda row: (row["start"], row["segment_id"]))

    for target in active_tracklets:
        if target["start_count"] > low_point_threshold:
            continue

        if max_target_length > 0 and target["length"] > max_target_length:
            continue

        best = None

        for source in active_tracklets:
            if source["segment_id"] == target["segment_id"]:
                continue

            if cross_id_only and source["track_id"] == target["track_id"]:
                continue

            if source["length"] < min_source_length:
                continue

            gap = target["start"] - source["end"]

            if gap <= 0 or gap > max_gap:
                continue

            if has_frame_overlap(source, target):
                continue

            predicted_xy = source["end_xy"] + source["velocity"] * gap
            distance = float(np.linalg.norm(target["start_xy"] - predicted_xy))

            if distance > max_distance:
                continue

            velocity_diff = float(np.linalg.norm(source["velocity"] - target["velocity"]))

            if max_velocity_diff > 0 and velocity_diff > max_velocity_diff:
                continue

            cost = (
                distance
                + 0.08 * gap
                + 0.02 * max(target["length"] - 1, 0)
                + velocity_weight * velocity_diff
            )

            if max_cost > 0 and cost > max_cost:
                continue

            if best is None or cost < best["cost"]:
                best = {
                    "source_segment_id": source["segment_id"],
                    "target_segment_id": target["segment_id"],
                    "source_id": source["track_id"],
                    "target_id": target["track_id"],
                    "source_segment_index": source["segment_index"],
                    "target_segment_index": target["segment_index"],
                    "source_end": source["end"],
                    "target_start": target["start"],
                    "gap": gap,
                    "distance": round(distance, 4),
                    "velocity_diff": round(velocity_diff, 4),
                    "target_start_count": target["start_count"],
                    "target_length": target["length"],
                    "cost": round(cost, 4),
                }

        if best is not None:
            pairs.append(best)

    pairs.sort(key=lambda row: (row["target_start"], row["cost"]))
    return pairs


def resolve_mapping(pairs: list[dict], tracklets: dict[int, dict]) -> dict[int, int]:
    segment_to_label = {}
    used_targets = set()

    for pair in pairs:
        target_segment_id = int(pair["target_segment_id"])

        if target_segment_id in used_targets:
            continue

        source_segment_id = int(pair["source_segment_id"])
        source_label = segment_to_label.get(source_segment_id, int(tracklets[source_segment_id]["track_id"]))

        if int(tracklets[target_segment_id]["track_id"]) == source_label:
            continue

        segment_to_label[target_segment_id] = source_label
        used_targets.add(target_segment_id)

    return segment_to_label


def apply_mapping(
    masks: dict[int, np.ndarray],
    mapping: dict[int, int],
    tracklets: dict[int, dict],
) -> dict[int, np.ndarray]:
    out = {frame_id: mask.copy() for frame_id, mask in masks.items()}

    for segment_id, new_id in mapping.items():
        segment = tracklets[segment_id]
        old_id = int(segment["track_id"])

        for obs in segment["observations"]:
            frame_id = int(obs["frame"])

            if frame_id in out:
                out[frame_id][masks[frame_id] == old_id] = int(new_id)

    return out


def write_pairs(path: Path, pairs: list[dict]) -> None:
    if not pairs:
        path.write_text("")
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline low-point tracklet smoothing for result.csv.")
    parser.add_argument("--input", required=True, help="Input result.csv.")
    parser.add_argument("--output", required=True, help="Smoothed output result.csv.")
    parser.add_argument("--merge-report", default=None)
    parser.add_argument("--max-gap", type=int, default=8)
    parser.add_argument("--max-distance", type=float, default=2.5)
    parser.add_argument("--low-point-threshold", type=int, default=3)
    parser.add_argument("--min-source-length", type=int, default=2)
    parser.add_argument("--split-gap", type=int, default=0)
    parser.add_argument("--split-jump-distance", type=float, default=0.0)
    parser.add_argument("--max-cost", type=float, default=0.0, help="0 disables this filter.")
    parser.add_argument("--max-target-length", type=int, default=0, help="0 disables this filter.")
    parser.add_argument("--max-velocity-diff", type=float, default=0.0, help="0 disables this filter.")
    parser.add_argument("--velocity-weight", type=float, default=0.0)
    parser.add_argument("--cross-id-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    masks, coords = load_result_csv(args.input)
    tracklets = build_tracklets(
        masks,
        coords,
        split_gap=args.split_gap,
        split_jump_distance=args.split_jump_distance,
    )
    pairs = find_merge_pairs(
        tracklets,
        max_gap=args.max_gap,
        max_distance=args.max_distance,
        low_point_threshold=args.low_point_threshold,
        min_source_length=args.min_source_length,
        max_cost=args.max_cost,
        max_target_length=args.max_target_length,
        max_velocity_diff=args.max_velocity_diff,
        velocity_weight=args.velocity_weight,
        cross_id_only=args.cross_id_only,
    )
    mapping = resolve_mapping(pairs, tracklets)
    smoothed = apply_mapping(masks, mapping, tracklets)
    write_result_csv(args.output, smoothed, coords)

    if args.merge_report is not None:
        write_pairs(Path(args.merge_report), pairs)

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"tracklets: {len(tracklets)}")
    print(f"candidate_pairs: {len(pairs)}")
    print(f"merged_segments: {len(mapping)}")
    print("[DONE] smoothed result csv")
