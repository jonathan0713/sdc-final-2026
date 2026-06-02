from __future__ import annotations

from collections import deque

import numpy as np


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

        matched, unmatched_detections, unmatched_tracks = self.match_tracks(detections)

        for track_idx, detection_idx in matched:
            track = self.tracks[track_idx]
            detection = detections[detection_idx]

            track["mean"] = detection[:2]
            track["hits"] += 1
            track["age"] = 0
            track["state"] = "confirmed" if track["hits"] >= self.min_hits else "tentative"
            track["det"] = detection
            track["prev_states"].append(detection[:2])

        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]

            self.tracks.append(
                {
                    "mean": detection[:2],
                    "track_id": self.next_track_id,
                    "hits": 1,
                    "age": 0,
                    "state": "tentative",
                    "det": detection,
                    "prev_states": deque(maxlen=MAX_TRAJ),
                }
            )
            self.next_track_id += 1

        for track_idx in unmatched_tracks:
            if track_idx >= len(self.tracks):
                continue

            track = self.tracks[track_idx]
            track["age"] += 1

            if track["age"] > self.max_age:
                track["state"] = "deleted"

        self.tracks = [t for t in self.tracks if t["state"] != "deleted"]

        confirmed_tracks = []

        for track in self.tracks:
            if track["state"] != "confirmed":
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
        #
        # Implement track-to-detection association.
        #
        # You may use any reasonable strategy, such as nearest-neighbor matching,
        # global assignment, motion prediction, gating, or a custom cost function.
        #
        # The output must follow the format described above.
        #
        # ================================ TODO: Implementation Ends Here ================================

        return matched, unmatched_detections, unmatched_tracks
