from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

from sdc_tracking_utils import (
    list_frame_names,
    load_radar_and_mask_data,
    parse_tracking_csv,
)


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


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
            "Run main.py first or pass --result explicitly."
        )

    return candidates[0]


def find_image_path(image_dir: Path, mask_name: str, frame_id: int) -> Path | None:
    stem_candidates = [Path(mask_name).stem, f"{frame_id:04d}"]

    for stem in stem_candidates:
        for extension in IMAGE_EXTENSIONS:
            image_path = image_dir / f"{stem}{extension}"
            if image_path.exists():
                return image_path

    return None


def color_for_track(track_id: int):
    cmap = plt.get_cmap("tab20")
    return cmap(int(track_id) % 20)


def build_track_centroid_cache(
    tracking_rows: dict[int, np.ndarray],
    mask_dir: Path,
    radar_dir: Path,
    frame_names: list[str],
) -> list[dict[int, tuple[float, float]]]:
    centroids_by_frame: list[dict[int, tuple[float, float]]] = []

    for frame_id, mask_name in enumerate(frame_names):
        _, radar_pc = load_radar_and_mask_data(mask_dir, radar_dir, mask_name)
        tracking_mask = tracking_rows.get(frame_id)

        if tracking_mask is None:
            centroids_by_frame.append({})
            continue

        frame_centroids = {}
        for track_id in np.unique(tracking_mask[tracking_mask >= 0]):
            point_indices = tracking_mask == track_id
            points = radar_pc[point_indices]

            if len(points) == 0:
                continue

            centroid = points[:, :2].mean(axis=0)
            frame_centroids[int(track_id)] = (float(centroid[0]), float(centroid[1]))

        centroids_by_frame.append(frame_centroids)

    return centroids_by_frame


class TrackingDebugViewer:
    def __init__(
        self,
        data_root: Path,
        seq: str,
        result_csv: Path,
        fps: float,
        start: int,
        end: int | None,
        stride: int,
        tail: int,
    ):
        self.data_root = data_root
        self.seq = seq
        self.result_csv = result_csv
        self.fps = fps
        self.tail = tail
        self.paused = False

        self.seq_dir = self.data_root / self.seq
        self.mask_dir = self.seq_dir / "mask_cluster"
        self.radar_dir = self.seq_dir / "radar"
        self.image_dir = self.seq_dir / "image"

        if not self.mask_dir.exists() or not self.radar_dir.exists():
            raise FileNotFoundError(f"Missing radar/ or mask_cluster/ under {self.seq_dir}")

        self.frame_names = list_frame_names(self.mask_dir)
        self.tracking_rows = parse_tracking_csv(self.result_csv)
        self.centroids_by_frame = build_track_centroid_cache(
            self.tracking_rows,
            self.mask_dir,
            self.radar_dir,
            self.frame_names,
        )

        last_frame = len(self.frame_names) - 1 if end is None else min(end, len(self.frame_names) - 1)
        self.frame_ids = list(range(max(start, 0), last_frame + 1, max(stride, 1)))
        if not self.frame_ids:
            raise ValueError("No frames selected. Check --start, --end, and --stride.")

        self.frame_cursor = 0
        self.fig, (self.camera_ax, self.radar_ax) = plt.subplots(
            1,
            2,
            figsize=(15, 7),
            gridspec_kw={"width_ratios": [1.25, 1.0]},
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def on_key_press(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "right":
            self.step(1)
        elif event.key == "left":
            self.step(-1)
        elif event.key == "home":
            self.frame_cursor = 0
            self.draw_current_frame()
        elif event.key == "end":
            self.frame_cursor = len(self.frame_ids) - 1
            self.draw_current_frame()
        elif event.key in {"q", "escape"}:
            plt.close(self.fig)

    def step(self, direction: int):
        self.frame_cursor = int(np.clip(self.frame_cursor + direction, 0, len(self.frame_ids) - 1))
        self.draw_current_frame()

    def update_animation(self, _frame_number):
        if not self.paused:
            self.frame_cursor = (self.frame_cursor + 1) % len(self.frame_ids)
            self.draw_current_frame()

    def draw_current_frame(self):
        frame_id = self.frame_ids[self.frame_cursor]
        mask_name = self.frame_names[frame_id]
        original_mask, radar_pc = load_radar_and_mask_data(self.mask_dir, self.radar_dir, mask_name)
        tracking_mask = self.tracking_rows.get(frame_id)

        if tracking_mask is None:
            tracking_mask = np.full(len(radar_pc), -2, dtype=np.int32)

        self.camera_ax.clear()
        self.radar_ax.clear()

        self.draw_camera(frame_id, mask_name)
        self.draw_radar(frame_id, radar_pc, original_mask, tracking_mask)

        status = "paused" if self.paused else "playing"
        self.fig.suptitle(
            f"{self.seq} | frame {frame_id:04d} | {status} | "
            "Space: pause/play, Left/Right: step, Home/End: jump, Q: quit",
            fontsize=11,
        )
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def draw_camera(self, frame_id: int, mask_name: str):
        image_path = find_image_path(self.image_dir, mask_name, frame_id)

        if image_path is None:
            self.camera_ax.text(0.5, 0.5, "No camera image", ha="center", va="center", fontsize=14)
            self.camera_ax.set_axis_off()
            return

        image = mpimg.imread(image_path)
        self.camera_ax.imshow(image)
        self.camera_ax.set_title(f"Camera image | {image_path.name}")
        self.camera_ax.set_axis_off()

    def draw_radar(
        self,
        frame_id: int,
        radar_pc: np.ndarray,
        original_mask: np.ndarray,
        tracking_mask: np.ndarray,
    ):
        static_indices = original_mask == -1
        tentative_indices = tracking_mask == -2
        confirmed_indices = tracking_mask >= 0

        if static_indices.any():
            self.radar_ax.scatter(
                -radar_pc[static_indices, 1],
                radar_pc[static_indices, 0],
                s=7,
                c="0.78",
                alpha=0.55,
                label="static",
            )

        if tentative_indices.any():
            self.radar_ax.scatter(
                -radar_pc[tentative_indices, 1],
                radar_pc[tentative_indices, 0],
                s=20,
                c="#f59e0b",
                alpha=0.85,
                label="moving unconfirmed",
            )

        for track_id in np.unique(tracking_mask[confirmed_indices]):
            track_id = int(track_id)
            point_indices = tracking_mask == track_id
            points = radar_pc[point_indices]
            color = color_for_track(track_id)

            self.radar_ax.scatter(
                -points[:, 1],
                points[:, 0],
                s=24,
                color=color,
                alpha=0.9,
            )

            centroid = points[:, :2].mean(axis=0)
            self.radar_ax.text(
                -centroid[1],
                centroid[0],
                str(track_id),
                color="black",
                fontsize=9,
                weight="bold",
                bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.8, "pad": 1.8},
            )

            history = []
            first_history_frame = max(0, frame_id - self.tail)
            for history_frame_id in range(first_history_frame, frame_id + 1):
                xy = self.centroids_by_frame[history_frame_id].get(track_id)
                if xy is not None:
                    history.append(xy)

            if len(history) > 1:
                history_points = np.asarray(history)
                self.radar_ax.plot(
                    -history_points[:, 1],
                    history_points[:, 0],
                    color=color,
                    linewidth=2.0,
                    alpha=0.9,
                )

        self.radar_ax.set_xlim(-60, 60)
        self.radar_ax.set_ylim(0, 85)
        self.radar_ax.set_aspect("equal", adjustable="box")
        self.radar_ax.set_xlabel("Lateral position (-Y)")
        self.radar_ax.set_ylabel("Forward position (X)")
        self.radar_ax.set_title("Radar bird's-eye tracking")
        self.radar_ax.grid(True, alpha=0.25)
        self.radar_ax.legend(loc="upper right", fontsize=8)

    def show(self):
        self.draw_current_frame()
        animation = FuncAnimation(
            self.fig,
            self.update_animation,
            interval=max(1, int(1000 / self.fps)),
            cache_frame_data=False,
        )
        plt.show()
        return animation

    def save_mp4(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def draw_frame_for_export(_frame_number):
            self.frame_cursor = _frame_number % len(self.frame_ids)
            self.draw_current_frame()

        animation = FuncAnimation(
            self.fig,
            draw_frame_for_export,
            frames=len(self.frame_ids),
            interval=max(1, int(1000 / self.fps)),
            cache_frame_data=False,
        )
        writer = FFMpegWriter(fps=self.fps)
        animation.save(output_path, writer=writer, dpi=120)
        print(f"[DONE] saved mp4: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive camera + radar tracking debug viewer.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seq", required=True, choices=["seq_1", "seq_2", "seq_3", "seq_4"])
    parser.add_argument("--result", default=None, help="Path to result.csv. Defaults to latest output for --seq.")
    parser.add_argument("--output-root", default="../outputs", help="Used only when --result is omitted.")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--tail", type=int, default=30, help="Number of previous frames used for trajectory tails.")
    parser.add_argument("--save-mp4", default=None, help="Optional path to save an MP4 instead of only showing a popup.")
    parser.add_argument("--no-popup", action="store_true", help="Do not open a popup window after saving MP4.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result_csv = Path(args.result) if args.result else find_latest_result(args.seq, args.output_root)
    print(f"result csv: {result_csv}")

    viewer = TrackingDebugViewer(
        data_root=Path(args.data_root),
        seq=args.seq,
        result_csv=result_csv,
        fps=args.fps,
        start=args.start,
        end=args.end,
        stride=args.stride,
        tail=args.tail,
    )

    if args.save_mp4:
        viewer.save_mp4(Path(args.save_mp4))

    if not args.no_popup:
        viewer.show()
