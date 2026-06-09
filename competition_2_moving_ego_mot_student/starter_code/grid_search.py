from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stdout
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter

from evaluate_tracking import evaluate
from main import run_sequence


def infer_gt_path(data_root: str | Path, seq: str) -> Path:
    seq_number = seq.split("_")[-1]
    gt_path = Path(data_root) / seq / f"gt_answer_seq{seq_number}.csv"

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Cannot infer GT path: {gt_path}. Pass --gt explicitly for this sequence."
        )

    return gt_path


def write_summary(summary_path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["MOTA_style_score"]),
            int(row["IDSW_proxy"]),
            int(row["fragmentation_proxy"]),
            int(row["FN_points"]),
            int(row["FP_points"]),
        ),
    )


def run_grid_search(
    data_root: str,
    seq: str,
    gt: str,
    output_root: str,
    max_distances: list[float],
    max_ages: list[int],
    min_hits_values: list[int],
    tracker_modes: list[str],
    ego_modes: list[str],
    verbose_runs: bool,
):
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_root = Path(output_root) / f"{seq}_grid_search_{started_at}"
    summary_path = grid_root / "grid_search_summary.csv"

    rows = []
    combinations = list(product(tracker_modes, ego_modes, max_distances, max_ages, min_hits_values))

    print("============================================================")
    print("Run tracker grid search")
    print("============================================================")
    print(f"seq: {seq}")
    print(f"data_root: {data_root}")
    print(f"gt: {gt}")
    print(f"grid_root: {grid_root}")
    print(f"combinations: {len(combinations)}")
    print(f"max_distances: {max_distances}")
    print(f"max_ages: {max_ages}")
    print(f"min_hits: {min_hits_values}")
    print(f"tracker_modes: {tracker_modes}")
    print(f"ego_modes: {ego_modes}")

    for combo_idx, (tracker_mode, ego_mode, max_distance, max_age, min_hits) in enumerate(combinations, start=1):
        combo_name = f"{tracker_mode}_{ego_mode}_md_{max_distance:g}_age_{max_age}_hits_{min_hits}".replace(".", "p")
        combo_root = grid_root / combo_name

        print("------------------------------------------------------------")
        print(
            f"[{combo_idx}/{len(combinations)}] "
            f"tracker_mode={tracker_mode}, ego_mode={ego_mode}, max_distance={max_distance}, "
            f"max_age={max_age}, min_hits={min_hits}"
        )

        t0 = perf_counter()
        combo_root.mkdir(parents=True, exist_ok=True)
        run_log = combo_root / "run.log"

        if verbose_runs:
            output_folder = run_sequence(
                data_root=data_root,
                seq=seq,
                output_root=str(combo_root),
                save_csv=True,
                plot_every=0,
                max_age=max_age,
                min_hits=min_hits,
                max_distance=max_distance,
                tracker_mode=tracker_mode,
                ego_mode=ego_mode,
            )
        else:
            with run_log.open("w") as f, redirect_stdout(f):
                output_folder = run_sequence(
                    data_root=data_root,
                    seq=seq,
                    output_root=str(combo_root),
                    save_csv=True,
                    plot_every=0,
                    max_age=max_age,
                    min_hits=min_hits,
                    max_distance=max_distance,
                    tracker_mode=tracker_mode,
                    ego_mode=ego_mode,
                )

        pred_csv = output_folder / "result.csv"
        metrics = evaluate(gt, pred_csv)
        elapsed_sec = perf_counter() - t0

        row = {
            "rank": 0,
            "seq": seq,
            "tracker_mode": tracker_mode,
            "ego_mode": ego_mode,
            "max_distance": max_distance,
            "max_age": max_age,
            "min_hits": min_hits,
            **metrics,
            "elapsed_sec": round(elapsed_sec, 3),
            "run_log": str(run_log),
            "result_csv": str(pred_csv),
            "output_folder": str(output_folder),
        }
        rows.append(row)
        rows = sort_rows(rows)

        for rank, ranked_row in enumerate(rows, start=1):
            ranked_row["rank"] = rank

        write_summary(summary_path, rows)

        best = rows[0]
        print(
            "[BEST] "
            f"score={best['MOTA_style_score']:.6f}, "
            f"tracker_mode={best['tracker_mode']}, "
            f"ego_mode={best['ego_mode']}, "
            f"max_distance={best['max_distance']}, "
            f"max_age={best['max_age']}, "
            f"min_hits={best['min_hits']}, "
            f"IDSW={best['IDSW_proxy']}, "
            f"Frag={best['fragmentation_proxy']}"
        )

    print("============================================================")
    print("[DONE] grid search complete")
    print(f"summary: {summary_path}")
    print("top 5:")

    for row in rows[:5]:
        print(
            f"  #{row['rank']}: score={row['MOTA_style_score']:.6f}, "
            f"tracker_mode={row['tracker_mode']}, "
            f"ego_mode={row['ego_mode']}, "
            f"max_distance={row['max_distance']}, max_age={row['max_age']}, "
            f"min_hits={row['min_hits']}, IDSW={row['IDSW_proxy']}, "
            f"Frag={row['fragmentation_proxy']}"
        )

    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search tracker hyperparameters on a public dev sequence.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seq", required=True, choices=["seq_1", "seq_3"])
    parser.add_argument("--gt", default=None, help="Defaults to <data-root>/<seq>/gt_answer_seqN.csv.")
    parser.add_argument("--output-root", default="../outputs")
    parser.add_argument("--max-distances", nargs="+", type=float, default=[3.0, 4.0, 5.0, 6.0, 7.0])
    parser.add_argument("--max-ages", nargs="+", type=int, default=[3, 5, 8, 10])
    parser.add_argument("--min-hits", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--tracker-modes",
        nargs="+",
        choices=["baseline", "kalman", "kalman_feature", "kalman_rr", "kalman_reid", "kalman_feature_reid", "kalman_low_point_reid"],
        default=["baseline"],
    )
    parser.add_argument(
        "--ego-modes",
        nargs="+",
        choices=["none", "add_xy", "add_x_neg_y", "add_yx", "add_y_neg_x"],
        default=["none"],
    )
    parser.add_argument("--verbose-runs", action="store_true", help="Print per-frame main.py logs to terminal.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gt = Path(args.gt) if args.gt else infer_gt_path(args.data_root, args.seq)

    run_grid_search(
        data_root=args.data_root,
        seq=args.seq,
        gt=str(gt),
        output_root=args.output_root,
        max_distances=args.max_distances,
        max_ages=args.max_ages,
        min_hits_values=args.min_hits,
        tracker_modes=args.tracker_modes,
        ego_modes=args.ego_modes,
        verbose_runs=args.verbose_runs,
    )
