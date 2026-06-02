from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sdc_tracker import Tracker
from sdc_tracking_utils import (
    ego_pos_distance,
    list_frame_names,
    load_radar_and_mask_data,
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
    #
    # Convert each moving cluster into a detection-level representation.
    #
    # Suggested logic:
    #   1. Find all unique valid cluster ids in moving_mask.
    #      Use cluster_id >= 0 only. Negative labels such as -1 / -2
    #      should not be converted into trackable detections.
    #   2. For each valid cluster id:
    #        - select its radar points from moving_radar_pc
    #        - compute mean x, y, z as centroid
    #        - compute mean range_rate if available
    #        - store all cluster points
    #
    # Expected outputs:
    #   cluster_centroids[cluster_id] = np.array([x, y, z])
    #   cluster_velocities[cluster_id] = mean_range_rate
    #   cluster_points_dict[cluster_id] = points
    #
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
    #
    # Generate the dictionary needed by save_tracking_mask_to_csv().
    #
    # Suggested logic:
    #   1. Iterate over confirmed_tracks.
    #   2. Read cluster_id and track_id from each confirmed track.
    #   3. Store cluster_track_dict[cluster_id] = track_id.
    #
    # If you design a different track output format, modify this function.
    #
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
    output_img_folder = output_folder / "tracking_vis"
    output_img_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    # TODO-3
    # ================================ TODO: Implementation Starts Here ================================
    #
    # Optional:
    #   1. Plot static radar points in gray.
    #   2. Plot moving clusters.
    #   3. Plot confirmed track ids.
    #   4. Plot track trajectory history.
    #   5. Save the figure.
    #
    # This TODO does not affect result.csv if left empty.
    #
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
):
    """
    Build detection array for Tracker.update().

    Expected detection format:
        [x, y, z, cluster_id, range_rate]
    """
    detections = []

    # TODO-4
    # ================================ TODO: Implementation Starts Here ================================
    #
    # Convert cluster-level dictionaries into a numpy detection array.
    #
    # Suggested output format:
    #   detections = np.array([
    #       [x, y, z, cluster_id, range_rate],
    #       ...
    #   ])
    #
    # This format must match the tracker implementation.
    #
    # ================================ TODO: Implementation Ends Here ================================

    if len(detections) == 0:
        return np.empty((0, 5), dtype=float)

    return np.asarray(detections, dtype=float)


def run_sequence(
    data_root: str,
    seq: str,
    output_root: str,
    save_csv: bool,
    plot_every: int,
    max_age: int,
    min_hits: int,
    max_distance: float,
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
        print(f"ego_global_pos: {ego_pos_path}")
        print(f"ego motion steps loaded: {len(ego_motion)}")
        print(
            "note: ego motion is loaded for students to use, but no ego compensation "
            "is implemented in this starter code."
        )

    tracker = Tracker(
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
    print("note: this is incomplete starter code. Implement TODO-1 to TODO-6.")

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

        detections = build_detections(cluster_centroids, cluster_velocities)

        confirmed_tracks = np.empty((0, 5), dtype=float)

        # TODO-5
        # ================================ TODO: Implementation Starts Here ================================
        #
        # Update the tracker with current-frame detections.
        #
        # Suggested logic:
        #   confirmed_tracks = tracker.update(detections)
        #
        # Students may also replace Tracker with their own implementation.
        #
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
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-distance", type=float, default=5.0)

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
    )
