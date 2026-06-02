from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment


MAX_TRAJ = 5


class Tracker:
    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 3,
        max_distance: float = 5.0,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.next_track_id = 0
        self.tracks = []

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update the tracker with new detections.

        Expected detection format:
            [x, y, z, cluster_id, range_rate]

        Returned confirmed track format:
            [x, y, z, cluster_id, track_id]

        Note:
            This starter code does not implement the data association method.
            Students must complete match_tracks() or replace this tracker with
            their own implementation.
        """
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 5), dtype=float)

        for track in self.tracks:
            track["updated"] = False

        matched, unmatched_detections, unmatched_tracks = self.match_tracks(detections)

        for track_idx, detection_idx in matched:
            track = self.tracks[track_idx]
            detection = detections[detection_idx]

            previous_mean = np.asarray(track["mean"], dtype=float)
            current_mean = detection[:2].astype(float)

            track["velocity"] = current_mean - previous_mean
            track["mean"] = current_mean
            track["hits"] += 1
            track["age"] = 0
            track["state"] = "confirmed" if track["hits"] >= self.min_hits else "tentative"
            track["det"] = detection
            track["updated"] = True
            track["prev_states"].append(current_mean)

        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]

            state = "confirmed" if self.min_hits <= 1 else "tentative"
            prev_states = deque(maxlen=MAX_TRAJ)
            prev_states.append(detection[:2].astype(float))

            self.tracks.append(
                {
                    "mean": detection[:2].astype(float),
                    "velocity": np.zeros(2, dtype=float),
                    "track_id": self.next_track_id,
                    "hits": 1,
                    "age": 0,
                    "state": state,
                    "det": detection,
                    "updated": True,
                    "prev_states": prev_states,
                }
            )
            self.next_track_id += 1

        for track_idx in unmatched_tracks:
            if track_idx >= len(self.tracks):
                continue

            track = self.tracks[track_idx]
            track["age"] += 1
            track["mean"] = np.asarray(track["mean"], dtype=float) + np.asarray(
                track.get("velocity", np.zeros(2, dtype=float)),
                dtype=float,
            )

            if track["age"] > self.max_age:
                track["state"] = "deleted"

        self.tracks = [t for t in self.tracks if t["state"] != "deleted"]

        confirmed_tracks = []

        for track in self.tracks:
            if track["state"] != "confirmed" or not track.get("updated", False):
                continue

            det = track["det"]
            confirmed_tracks.append(
                [
                    float(track["mean"][0]),
                    float(track["mean"][1]),
                    float(det[2]),
                    int(det[3]),
                    int(track["track_id"]),
                ]
            )

        return np.asarray(confirmed_tracks, dtype=float)

    def match_tracks(self, detections: np.ndarray):
        """
        Associate current-frame detections with existing tracks.

        Required return values:
            matched:
                list of (track_index, detection_index)

            unmatched_detections:
                list of detection indices that are not assigned to any existing track

            unmatched_tracks:
                list of track indices that are not assigned to any current detection

        TODO-6:
            Implement your own data association strategy here.

        Important:
            This function currently returns no matches. Therefore, the starter
            code can run and produce a correctly formatted result.csv, but it
            will not produce meaningful tracking results until this TODO is
            completed.
        """
        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        # TODO-6
        # ================================ TODO: Implementation Starts Here ================================
        if len(detections) == 0 or len(self.tracks) == 0:
            return matched, unmatched_detections, unmatched_tracks

        cost = np.full((len(self.tracks), len(detections)), fill_value=1e6, dtype=float)

        for track_idx, track in enumerate(self.tracks):
            if track["state"] == "deleted":
                continue

            age = int(track.get("age", 0))
            mean = np.asarray(track["mean"], dtype=float)
            velocity = np.asarray(track.get("velocity", np.zeros(2, dtype=float)), dtype=float)
            predicted_xy = mean + velocity * max(age, 1)
            previous_det = np.asarray(track.get("det", np.zeros(5, dtype=float)), dtype=float)

            for detection_idx, detection in enumerate(detections):
                xy_distance = np.linalg.norm(detection[:2] - predicted_xy)

                if xy_distance > self.max_distance:
                    continue

                z_distance = abs(float(detection[2]) - float(previous_det[2]))
                rr_distance = abs(float(detection[4]) - float(previous_det[4]))

                cost[track_idx, detection_idx] = xy_distance + 0.1 * z_distance + 0.05 * rr_distance

        row_indices, col_indices = linear_sum_assignment(cost)

        matched = []
        matched_track_indices = set()
        matched_detection_indices = set()

        for track_idx, detection_idx in zip(row_indices, col_indices):
            if cost[track_idx, detection_idx] >= 1e6:
                continue

            matched.append((int(track_idx), int(detection_idx)))
            matched_track_indices.add(int(track_idx))
            matched_detection_indices.add(int(detection_idx))

        unmatched_detections = [
            detection_idx for detection_idx in range(len(detections)) if detection_idx not in matched_detection_indices
        ]
        unmatched_tracks = [
            track_idx
            for track_idx, track in enumerate(self.tracks)
            if track_idx not in matched_track_indices and track["state"] != "deleted"
        ]
        # ================================ TODO: Implementation Ends Here ================================

        return matched, unmatched_detections, unmatched_tracks
