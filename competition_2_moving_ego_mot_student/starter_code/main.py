from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from sdc_tracker import FeatureKalmanTracker, FeatureReIDKalmanTracker, KalmanTracker, LowPointReIDKalmanTracker, RangeRateKalmanTracker, ReIDKalmanTracker, Tracker
from sdc_tracking_utils import (
    ego_pos_distance,
    list_frame_names,
    load_radar_and_mask_data,
    read_ego_pos,
    save_tracking_mask_to_csv,
    separate_static_and_moving,
)


def calculate_cluster_centroids(
    moving_mask: np.ndarray,
    moving_radar_pc: np.ndarray,
):
    """
    Calculate cluster-level detections from moving radar points.

    Inputs:
        moving_mask:
            cluster ids for moving radar points

        moving_radar_pc:
            moving radar points with columns:
            [x, y, z, range_rate, rcs]

    Returns:
        cluster_centroids:
            dict[int, np.ndarray], cluster_id -> [x, y, z]

        cluster_velocities:
            dict[int, float], cluster_id -> mean range_rate

        cluster_points_dict:
            dict[int, np.ndarray], cluster_id -> points
    """
    cluster_centroids = {}
    cluster_velocities = {}
    cluster_points_dict = {}

    # TODO-1
    # ================================ TODO: Implementation Starts Here ================================
    valid_cluster_ids = np.unique(moving_mask[moving_mask >= 0])

    for cluster_id in valid_cluster_ids:
        point_indices = moving_mask == cluster_id
        points = moving_radar_pc[point_indices]

        if len(points) == 0:
            continue

        cluster_id = int(cluster_id)
        cluster_centroids[cluster_id] = points[:, :3].mean(axis=0)
        cluster_velocities[cluster_id] = float(points[:, 3].mean()) if points.shape[1] > 3 else 0.0
        cluster_points_dict[cluster_id] = points
    # ================================ TODO: Implementation Ends Here ================================

    return cluster_centroids, cluster_velocities, cluster_points_dict


def generate_cluster_track_dict(
    confirmed_tracks: np.ndarray,
    cluster_centroids: dict[int, np.ndarray],
    cluster_points_dict: dict[int, np.ndarray],
):
    """
    Convert confirmed tracks into cluster_id -> track_id mapping.

    Expected confirmed_tracks format from Tracker.update():
        [x, y, z, cluster_id, track_id]

    Returned format:
        {cluster_id: track_id}
    """
    cluster_track_dict = {}

    # TODO-2
    # ================================ TODO: Implementation Starts Here ================================
    for track in confirmed_tracks:
        if len(track) < 5:
            continue

        cluster_id = int(track[3])
        track_id = int(track[4])

        if cluster_id in cluster_centroids and cluster_id in cluster_points_dict:
            cluster_track_dict[cluster_id] = track_id
    # ================================ TODO: Implementation Ends Here ================================

    return cluster_track_dict


def visualize_tracking(
    static_radar_pc: np.ndarray,
    moving_radar_pc: np.ndarray,
    moving_mask: np.ndarray,
    confirmed_tracks: np.ndarray,
    track_trajectories: dict[int, list[tuple[float, float]]],
    frame_idx: int,
    output_folder: Path,
):
    """
    Optional visualization function.

    This function is not required for Kaggle submission.
    Students may modify it to inspect tracking behavior.
    """
    import matplotlib.pyplot as plt

    output_img_folder = output_folder / "tracking_vis"
    output_img_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    # TODO-3
    # ================================ TODO: Implementation Starts Here ================================
    if len(static_radar_pc) > 0:
        ax.scatter(-static_radar_pc[:, 1], static_radar_pc[:, 0], s=6, c="0.75", alpha=0.45, label="static")

    if len(moving_radar_pc) > 0:
        for cluster_id in np.unique(moving_mask[moving_mask >= 0]):
            points = moving_radar_pc[moving_mask == cluster_id]
            ax.scatter(-points[:, 1], points[:, 0], s=18, alpha=0.8)

    for track in confirmed_tracks:
        x, y, _z, _cluster_id, track_id = track
        track_id = int(track_id)
        track_trajectories.setdefault(track_id, []).append((float(x), float(y)))
        history = track_trajectories[track_id][-20:]

        if len(history) > 1:
            hist = np.asarray(history)
            ax.plot(-hist[:, 1], hist[:, 0], linewidth=1.5, alpha=0.9)

        ax.text(-y, x, str(track_id), fontsize=9, weight="bold")
    # ================================ TODO: Implementation Ends Here ================================

    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Lateral position (-Y)")
    ax.set_ylabel("Forward position (X)")
    ax.set_title(f"Tracking visualization | frame {frame_idx}")
    ax.grid(True, alpha=0.25)

    fig.savefig(output_img_folder / f"frame_{frame_idx:04d}.png", dpi=120)
    plt.close(fig)


def build_detections(
    cluster_centroids: dict[int, np.ndarray],
    cluster_velocities: dict[int, float],
    cluster_points_dict: dict[int, np.ndarray] | None = None,
):
    """
    Build detection array for Tracker.update().

    Expected detection format:
        [x, y, z, cluster_id, range_rate]
    """
    detections = []

    # TODO-4
    # ================================ TODO: Implementation Starts Here ================================
    for cluster_id in sorted(cluster_centroids):
        centroid = np.asarray(cluster_centroids[cluster_id], dtype=float)

        if centroid.shape[0] < 3:
            continue

        points = None if cluster_points_dict is None else cluster_points_dict.get(cluster_id)

        if points is not None and len(points) > 0:
            rcs_mean = float(points[:, 4].mean()) if points.shape[1] > 4 else 0.0
            point_count = float(len(points))
            extents = np.ptp(points[:, :3], axis=0)
            rr_std = float(points[:, 3].std()) if points.shape[1] > 3 else 0.0
            rcs_std = float(points[:, 4].std()) if points.shape[1] > 4 else 0.0
        else:
            rcs_mean = 0.0
            point_count = 0.0
            extents = np.zeros(3, dtype=float)
            rr_std = 0.0
            rcs_std = 0.0

        detections.append(
            [
                float(centroid[0]),
                float(centroid[1]),
                float(centroid[2]),
                int(cluster_id),
                float(cluster_velocities.get(cluster_id, 0.0)),
                rcs_mean,
                point_count,
                float(extents[0]),
                float(extents[1]),
                float(extents[2]),
                rr_std,
                rcs_std,
            ]
        )
    # ================================ TODO: Implementation Ends Here ================================

    if len(detections) == 0:
        return np.empty((0, 5), dtype=float)

    return np.asarray(detections, dtype=float)


def apply_ego_translation_compensation(
    detections: np.ndarray,
    ego_positions: np.ndarray,
    frame_idx: int,
    ego_mode: str,
) -> np.ndarray:
    if ego_mode == "none" or len(detections) == 0:
        return detections

    if frame_idx >= len(ego_positions):
        return detections

    compensated = detections.copy()
    ego_x, ego_y = ego_positions[frame_idx, :2]

    if ego_mode == "add_xy":
        offset = np.asarray([ego_x, ego_y], dtype=float)
    elif ego_mode == "add_x_neg_y":
        offset = np.asarray([ego_x, -ego_y], dtype=float)
    elif ego_mode == "add_yx":
        offset = np.asarray([ego_y, ego_x], dtype=float)
    elif ego_mode == "add_y_neg_x":
        offset = np.asarray([ego_y, -ego_x], dtype=float)
    else:
        raise ValueError(f"Unknown ego_mode: {ego_mode}")

    compensated[:, :2] = compensated[:, :2] + offset
    return compensated


def run_sequence(
    data_root: str,
    seq: str,
    output_root: str,
    save_csv: bool,
    plot_every: int,
    max_age: int,
    min_hits: int,
    max_distance: float,
    tracker_mode: str = "baseline",
    ego_mode: str = "none",
):
    data_root = Path(data_root)
    seq_dir = data_root / seq

    mask_dir = seq_dir / "mask_cluster"
    radar_dir = seq_dir / "radar"

    if not mask_dir.exists() or not radar_dir.exists():
        raise FileNotFoundError(f"Missing radar/ or mask_cluster/ under {seq_dir}")

    output_root = Path(output_root)
    output_folder = output_root / f"{seq}_student_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_folder.mkdir(parents=True, exist_ok=True)

    result_csv = output_folder / "result.csv"

    if result_csv.exists():
        result_csv.unlink()

    if seq in ["seq_3", "seq_4"]:
        ego_pos_path = seq_dir / "ego_global_pos.txt"
        ego_motion = ego_pos_distance(ego_pos_path)
        ego_positions = read_ego_pos(ego_pos_path)
        print(f"ego_global_pos: {ego_pos_path}")
        print(f"ego motion steps loaded: {len(ego_motion)}")
        print(
            "note: ego motion is loaded; the current baseline uses radar-cluster "
            "association without explicit ego compensation."
        )
    else:
        ego_positions = np.zeros((0, 3), dtype=float)

    tracker_classes = {
        "baseline": Tracker,
        "kalman": KalmanTracker,
        "kalman_feature": FeatureKalmanTracker,
        "kalman_rr": RangeRateKalmanTracker,
        "kalman_reid": ReIDKalmanTracker,
        "kalman_feature_reid": FeatureReIDKalmanTracker,
        "kalman_low_point_reid": LowPointReIDKalmanTracker,
    }
    tracker_cls = tracker_classes[tracker_mode]
    tracker = tracker_cls(
        max_age=max_age,
        min_hits=min_hits,
        max_distance=max_distance,
    )

    track_trajectories = {}
    frame_names = list_frame_names(mask_dir)

    print("============================================================")
    print("Run student tracking starter")
    print("============================================================")
    print(f"seq: {seq}")
    print(f"frames: {len(frame_names)}")
    print(f"output: {output_folder}")
    print(f"max_age: {max_age}")
    print(f"min_hits: {min_hits}")
    print(f"max_distance: {max_distance}")
    print(f"tracker_mode: {tracker_mode}")
    print(f"ego_mode: {ego_mode}")
    print("note: cluster-level tracking implementation is active.")

    for frame_idx, mask_name in enumerate(frame_names):
        mask, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, mask_name)

        moving_mask, static_mask, moving_radar_pc, static_radar_pc, moving_indices = separate_static_and_moving(
            mask,
            radar_pc,
        )

        cluster_centroids, cluster_velocities, cluster_points_dict = calculate_cluster_centroids(
            moving_mask,
            moving_radar_pc,
        )

        detections = build_detections(cluster_centroids, cluster_velocities, cluster_points_dict)
        tracker_detections = apply_ego_translation_compensation(
            detections=detections,
            ego_positions=ego_positions,
            frame_idx=frame_idx,
            ego_mode=ego_mode,
        )

        confirmed_tracks = np.empty((0, 5), dtype=float)

        # TODO-5
        # ================================ TODO: Implementation Starts Here ================================
        confirmed_tracks = tracker.update(tracker_detections)
        # ================================ TODO: Implementation Ends Here ================================

        if save_csv:
            cluster_track_dict = generate_cluster_track_dict(
                confirmed_tracks,
                cluster_centroids,
                cluster_points_dict,
            )

            save_tracking_mask_to_csv(
                csv_path=result_csv,
                frame_id=frame_idx,
                mask=mask,
                cluster_track_dict=cluster_track_dict,
                moving_indices=moving_indices,
                radar_points=radar_pc,
            )

        if plot_every > 0 and frame_idx % plot_every == 0:
            visualize_tracking(
                static_radar_pc=static_radar_pc,
                moving_radar_pc=moving_radar_pc,
                moving_mask=moving_mask,
                confirmed_tracks=confirmed_tracks,
                track_trajectories=track_trajectories,
                frame_idx=frame_idx,
                output_folder=output_folder,
            )

        if frame_idx % 50 == 0:
            print(
                f"  frame {frame_idx:04d}: "
                f"detections={len(detections)}, confirmed={len(confirmed_tracks)}"
            )

    print(f"[DONE] output folder: {output_folder}")

    if save_csv:
        print(f"[DONE] result csv: {result_csv}")

    return output_folder


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seq", required=True, choices=["seq_1", "seq_2", "seq_3", "seq_4"])
    parser.add_argument("--output-root", required=True)

    parser.add_argument(
        "--no-save-csv",
        action="store_true",
        help="Disable writing result.csv. By default, result.csv is always saved.",
    )
    parser.add_argument("--plot-every", type=int, default=0)

    parser.add_argument("--max-age", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--max-distance", type=float, default=7.0)
    parser.add_argument(
        "--tracker-mode",
        choices=["baseline", "kalman", "kalman_feature", "kalman_rr", "kalman_reid", "kalman_feature_reid", "kalman_low_point_reid"],
        default="baseline",
    )
    parser.add_argument(
        "--ego-mode",
        choices=["none", "add_xy", "add_x_neg_y", "add_yx", "add_y_neg_x"],
        default="none",
        help="Track in an approximate ego-translation-compensated coordinate frame.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_sequence(
        data_root=args.data_root,
        seq=args.seq,
        output_root=args.output_root,
        save_csv=not args.no_save_csv,
        plot_every=args.plot_every,
        max_age=args.max_age,
        min_hits=args.min_hits,
        max_distance=args.max_distance,
        tracker_mode=args.tracker_mode,
        ego_mode=args.ego_mode,
    )
