from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment


MAX_TRAJ = 5
FEATURE_DIM = 7
FEATURE_START = 5
POINT_COUNT_INDEX = 6
LOW_POINT_THRESHOLD = 3.0


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


class KalmanTracker:
    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 1,
        max_distance: float = 5.0,
        process_noise: float = 1.0,
        measurement_noise: float = 2.0,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.next_track_id = 0
        self.tracks = []

        self.f = np.asarray(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.h = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        self.q = np.eye(4, dtype=float) * process_noise
        self.r = np.eye(2, dtype=float) * measurement_noise

    def update(self, detections: np.ndarray) -> np.ndarray:
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 5), dtype=float)

        for track in self.tracks:
            track["updated"] = False
            self.predict_track(track)

        matched, unmatched_detections, unmatched_tracks = self.match_tracks(detections)

        for track_idx, detection_idx in matched:
            track = self.tracks[track_idx]
            detection = detections[detection_idx]
            self.update_track(track, detection[:2])

            track["hits"] += 1
            track["age"] = 0
            track["state"] = "confirmed" if track["hits"] >= self.min_hits else "tentative"
            track["det"] = detection
            track["updated"] = True
            track["prev_states"].append(track["x"][:2].copy())

        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            state = "confirmed" if self.min_hits <= 1 else "tentative"
            prev_states = deque(maxlen=MAX_TRAJ)
            x = np.asarray([detection[0], detection[1], 0.0, 0.0], dtype=float)
            prev_states.append(x[:2].copy())

            self.tracks.append(
                {
                    "x": x,
                    "p": np.diag([4.0, 4.0, 25.0, 25.0]).astype(float),
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

            if track["age"] > self.max_age:
                track["state"] = "deleted"

        self.tracks = [track for track in self.tracks if track["state"] != "deleted"]

        confirmed_tracks = []

        for track in self.tracks:
            if track["state"] != "confirmed" or not track.get("updated", False):
                continue

            det = track["det"]
            confirmed_tracks.append(
                [
                    float(track["x"][0]),
                    float(track["x"][1]),
                    float(det[2]),
                    int(det[3]),
                    int(track["track_id"]),
                ]
            )

        return np.asarray(confirmed_tracks, dtype=float)

    def predict_track(self, track: dict) -> None:
        track["x"] = self.f @ track["x"]
        track["p"] = self.f @ track["p"] @ self.f.T + self.q

    def update_track(self, track: dict, measurement_xy: np.ndarray) -> None:
        z = np.asarray(measurement_xy, dtype=float)
        innovation = z - self.h @ track["x"]
        s = self.h @ track["p"] @ self.h.T + self.r
        k = track["p"] @ self.h.T @ np.linalg.inv(s)
        track["x"] = track["x"] + k @ innovation
        identity = np.eye(4, dtype=float)
        track["p"] = (identity - k @ self.h) @ track["p"]

    def match_tracks(self, detections: np.ndarray):
        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(detections) == 0 or len(self.tracks) == 0:
            return matched, unmatched_detections, unmatched_tracks

        cost = np.full((len(self.tracks), len(detections)), fill_value=1e6, dtype=float)

        for track_idx, track in enumerate(self.tracks):
            if track["state"] == "deleted":
                continue

            predicted_xy = track["x"][:2]
            previous_det = np.asarray(track.get("det", np.zeros(5, dtype=float)), dtype=float)
            s = self.h @ track["p"] @ self.h.T + self.r
            s_inv = np.linalg.inv(s)

            for detection_idx, detection in enumerate(detections):
                innovation = detection[:2] - predicted_xy
                xy_distance = np.linalg.norm(innovation)

                if xy_distance > self.max_distance:
                    continue

                mahalanobis_distance = float(np.sqrt(innovation.T @ s_inv @ innovation))
                z_distance = abs(float(detection[2]) - float(previous_det[2]))
                rr_distance = abs(float(detection[4]) - float(previous_det[4]))

                cost[track_idx, detection_idx] = (
                    mahalanobis_distance
                    + 0.1 * xy_distance
                    + 0.1 * z_distance
                    + 0.05 * rr_distance
                )

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

        return matched, unmatched_detections, unmatched_tracks


def cluster_feature_cost(detection: np.ndarray, previous_det: np.ndarray) -> float:
    if len(detection) < FEATURE_START + FEATURE_DIM or len(previous_det) < FEATURE_START + FEATURE_DIM:
        return 0.0

    rcs_mean_diff = abs(float(detection[5]) - float(previous_det[5]))
    count_diff = abs(np.log1p(max(float(detection[6]), 0.0)) - np.log1p(max(float(previous_det[6]), 0.0)))
    extent_diff = np.linalg.norm(detection[7:10] - previous_det[7:10])
    rr_std_diff = abs(float(detection[10]) - float(previous_det[10]))
    rcs_std_diff = abs(float(detection[11]) - float(previous_det[11]))

    return (
        0.02 * rcs_mean_diff
        + 0.25 * count_diff
        + 0.08 * extent_diff
        + 0.05 * rr_std_diff
        + 0.01 * rcs_std_diff
    )


def detection_point_count(detection: np.ndarray) -> float:
    if len(detection) <= POINT_COUNT_INDEX:
        return 99.0

    return float(detection[POINT_COUNT_INDEX])


def is_low_point_detection(detection: np.ndarray, threshold: float = LOW_POINT_THRESHOLD) -> bool:
    return detection_point_count(detection) <= threshold


class FeatureKalmanTracker(KalmanTracker):
    def match_tracks(self, detections: np.ndarray):
        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(detections) == 0 or len(self.tracks) == 0:
            return matched, unmatched_detections, unmatched_tracks

        cost = np.full((len(self.tracks), len(detections)), fill_value=1e6, dtype=float)

        for track_idx, track in enumerate(self.tracks):
            if track["state"] == "deleted":
                continue

            predicted_xy = track["x"][:2]
            previous_det = np.asarray(track.get("det", np.zeros(FEATURE_START + FEATURE_DIM, dtype=float)), dtype=float)
            s = self.h @ track["p"] @ self.h.T + self.r
            s_inv = np.linalg.inv(s)

            for detection_idx, detection in enumerate(detections):
                innovation = detection[:2] - predicted_xy
                xy_distance = np.linalg.norm(innovation)

                if xy_distance > self.max_distance:
                    continue

                mahalanobis_distance = float(np.sqrt(innovation.T @ s_inv @ innovation))
                z_distance = abs(float(detection[2]) - float(previous_det[2]))
                rr_distance = abs(float(detection[4]) - float(previous_det[4]))

                cost[track_idx, detection_idx] = (
                    mahalanobis_distance
                    + 0.1 * xy_distance
                    + 0.1 * z_distance
                    + 0.05 * rr_distance
                    + cluster_feature_cost(detection, previous_det)
                )

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

        return matched, unmatched_detections, unmatched_tracks


class ReIDKalmanTracker(KalmanTracker):
    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 1,
        max_distance: float = 5.0,
        process_noise: float = 1.0,
        measurement_noise: float = 2.0,
        reid_max_age: int = 25,
    ):
        super().__init__(
            max_age=max_age,
            min_hits=min_hits,
            max_distance=max_distance,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
        self.reid_max_age = reid_max_age
        self.lost_tracks = []

    def update(self, detections: np.ndarray) -> np.ndarray:
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 5), dtype=float)

        for track in self.tracks:
            track["updated"] = False
            self.predict_track(track)

        for track in self.lost_tracks:
            self.predict_track(track)
            track["lost_age"] += 1

        self.lost_tracks = [track for track in self.lost_tracks if track["lost_age"] <= self.reid_max_age]

        matched, unmatched_detections, unmatched_tracks = self.match_tracks(detections)

        for track_idx, detection_idx in matched:
            track = self.tracks[track_idx]
            detection = detections[detection_idx]
            self.update_track(track, detection[:2])

            track["hits"] += 1
            track["age"] = 0
            track["state"] = "confirmed" if track["hits"] >= self.min_hits else "tentative"
            track["det"] = detection
            track["updated"] = True
            track["prev_states"].append(track["x"][:2].copy())

        deleted_tracks = []

        for track_idx in unmatched_tracks:
            if track_idx >= len(self.tracks):
                continue

            track = self.tracks[track_idx]
            track["age"] += 1

            if track["age"] > self.max_age:
                track["state"] = "deleted"
                if track["hits"] >= self.min_hits:
                    lost_track = track.copy()
                    lost_track["lost_age"] = 0
                    lost_track["updated"] = False
                    lost_track["state"] = "lost"
                    deleted_tracks.append(lost_track)

        self.lost_tracks.extend(deleted_tracks)
        self.tracks = [track for track in self.tracks if track["state"] != "deleted"]

        reid_matched, remaining_unmatched_detections, matched_lost_indices = self.match_lost_tracks(
            detections,
            unmatched_detections,
        )

        reactivated_tracks = []

        for lost_idx, detection_idx in reid_matched:
            lost_track = self.lost_tracks[lost_idx]
            detection = detections[detection_idx]
            self.update_track(lost_track, detection[:2])

            lost_track["hits"] += 1
            lost_track["age"] = 0
            lost_track["state"] = "confirmed"
            lost_track["det"] = detection
            lost_track["updated"] = True
            lost_track["prev_states"].append(lost_track["x"][:2].copy())
            lost_track.pop("lost_age", None)
            reactivated_tracks.append(lost_track)

        if matched_lost_indices:
            self.lost_tracks = [
                track for idx, track in enumerate(self.lost_tracks) if idx not in matched_lost_indices
            ]

        self.tracks.extend(reactivated_tracks)

        for detection_idx in remaining_unmatched_detections:
            detection = detections[detection_idx]
            state = "confirmed" if self.min_hits <= 1 else "tentative"
            prev_states = deque(maxlen=MAX_TRAJ)
            x = np.asarray([detection[0], detection[1], 0.0, 0.0], dtype=float)
            prev_states.append(x[:2].copy())

            self.tracks.append(
                {
                    "x": x,
                    "p": np.diag([4.0, 4.0, 25.0, 25.0]).astype(float),
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

        confirmed_tracks = []

        for track in self.tracks:
            if track["state"] != "confirmed" or not track.get("updated", False):
                continue

            det = track["det"]
            confirmed_tracks.append(
                [
                    float(track["x"][0]),
                    float(track["x"][1]),
                    float(det[2]),
                    int(det[3]),
                    int(track["track_id"]),
                ]
            )

        return np.asarray(confirmed_tracks, dtype=float)

    def match_lost_tracks(self, detections: np.ndarray, detection_indices: list[int]):
        if len(self.lost_tracks) == 0 or len(detection_indices) == 0:
            return [], detection_indices, set()

        cost = np.full((len(self.lost_tracks), len(detection_indices)), fill_value=1e6, dtype=float)

        for lost_idx, track in enumerate(self.lost_tracks):
            predicted_xy = track["x"][:2]
            previous_det = np.asarray(track.get("det", np.zeros(5, dtype=float)), dtype=float)
            s = self.h @ track["p"] @ self.h.T + self.r
            s_inv = np.linalg.inv(s)

            for col_idx, detection_idx in enumerate(detection_indices):
                detection = detections[detection_idx]
                innovation = detection[:2] - predicted_xy
                xy_distance = np.linalg.norm(innovation)

                if xy_distance > self.max_distance:
                    continue

                mahalanobis_distance = float(np.sqrt(innovation.T @ s_inv @ innovation))
                z_distance = abs(float(detection[2]) - float(previous_det[2]))
                rr_distance = abs(float(detection[4]) - float(previous_det[4]))
                lost_age = float(track.get("lost_age", 0))

                cost[lost_idx, col_idx] = (
                    mahalanobis_distance
                    + 0.1 * xy_distance
                    + 0.1 * z_distance
                    + 0.05 * rr_distance
                    + 0.03 * lost_age
                )

        row_indices, col_indices = linear_sum_assignment(cost)

        matched = []
        matched_lost_indices = set()
        matched_detection_indices = set()

        for lost_idx, col_idx in zip(row_indices, col_indices):
            if cost[lost_idx, col_idx] >= 1e6:
                continue

            detection_idx = int(detection_indices[col_idx])
            matched.append((int(lost_idx), detection_idx))
            matched_lost_indices.add(int(lost_idx))
            matched_detection_indices.add(detection_idx)

        remaining_unmatched_detections = [
            detection_idx for detection_idx in detection_indices if detection_idx not in matched_detection_indices
        ]

        return matched, remaining_unmatched_detections, matched_lost_indices


class LowPointReIDKalmanTracker(ReIDKalmanTracker):
    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 1,
        max_distance: float = 5.0,
        process_noise: float = 1.0,
        measurement_noise: float = 2.0,
        reid_max_age: int = 35,
        low_point_threshold: float = LOW_POINT_THRESHOLD,
        low_point_distance_scale: float = 1.0,
    ):
        super().__init__(
            max_age=max_age,
            min_hits=min_hits,
            max_distance=max_distance,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            reid_max_age=reid_max_age,
        )
        self.low_point_threshold = low_point_threshold
        self.low_point_distance_scale = low_point_distance_scale

    def low_point_gate(self, detection: np.ndarray) -> float:
        if is_low_point_detection(detection, self.low_point_threshold):
            return self.max_distance * self.low_point_distance_scale

        return self.max_distance

    def low_point_cost_adjustment(self, track: dict, detection: np.ndarray) -> float:
        if not is_low_point_detection(detection, self.low_point_threshold):
            return 0.0

        previous_det = np.asarray(track.get("det", np.zeros(FEATURE_START + FEATURE_DIM, dtype=float)), dtype=float)
        previous_count = detection_point_count(previous_det)
        track_age = float(track.get("age", 0))
        lost_age = float(track.get("lost_age", 0))

        adjustment = 0.0

        if previous_count <= self.low_point_threshold and track_age > 0:
            adjustment -= min(0.25, 0.08 * track_age)

        if previous_count <= self.low_point_threshold and 0 < lost_age <= 8:
            adjustment -= 0.20

        if previous_count > self.low_point_threshold and track_age == 0:
            adjustment += 0.45

        if previous_count > 5.0 and detection_point_count(detection) <= 1.0:
            adjustment += 0.25

        return adjustment

    def match_tracks(self, detections: np.ndarray):
        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(detections) == 0 or len(self.tracks) == 0:
            return matched, unmatched_detections, unmatched_tracks

        cost = np.full((len(self.tracks), len(detections)), fill_value=1e6, dtype=float)

        for track_idx, track in enumerate(self.tracks):
            if track["state"] == "deleted":
                continue

            predicted_xy = track["x"][:2]
            previous_det = np.asarray(track.get("det", np.zeros(5, dtype=float)), dtype=float)
            s = self.h @ track["p"] @ self.h.T + self.r
            s_inv = np.linalg.inv(s)

            for detection_idx, detection in enumerate(detections):
                innovation = detection[:2] - predicted_xy
                xy_distance = np.linalg.norm(innovation)

                if xy_distance > self.low_point_gate(detection):
                    continue

                mahalanobis_distance = float(np.sqrt(innovation.T @ s_inv @ innovation))
                z_distance = abs(float(detection[2]) - float(previous_det[2]))
                rr_distance = abs(float(detection[4]) - float(previous_det[4]))

                base_cost = (
                    mahalanobis_distance
                    + 0.1 * xy_distance
                    + 0.1 * z_distance
                    + 0.05 * rr_distance
                )
                cost[track_idx, detection_idx] = max(
                    0.0,
                    base_cost + self.low_point_cost_adjustment(track, detection),
                )

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

        return matched, unmatched_detections, unmatched_tracks

    def match_lost_tracks(self, detections: np.ndarray, detection_indices: list[int]):
        if len(self.lost_tracks) == 0 or len(detection_indices) == 0:
            return [], detection_indices, set()

        cost = np.full((len(self.lost_tracks), len(detection_indices)), fill_value=1e6, dtype=float)

        for lost_idx, track in enumerate(self.lost_tracks):
            predicted_xy = track["x"][:2]
            previous_det = np.asarray(track.get("det", np.zeros(5, dtype=float)), dtype=float)
            s = self.h @ track["p"] @ self.h.T + self.r
            s_inv = np.linalg.inv(s)

            for col_idx, detection_idx in enumerate(detection_indices):
                detection = detections[detection_idx]
                innovation = detection[:2] - predicted_xy
                xy_distance = np.linalg.norm(innovation)

                if xy_distance > self.low_point_gate(detection):
                    continue

                mahalanobis_distance = float(np.sqrt(innovation.T @ s_inv @ innovation))
                z_distance = abs(float(detection[2]) - float(previous_det[2]))
                rr_distance = abs(float(detection[4]) - float(previous_det[4]))
                lost_age = float(track.get("lost_age", 0))
                base_cost = (
                    mahalanobis_distance
                    + 0.1 * xy_distance
                    + 0.1 * z_distance
                    + 0.05 * rr_distance
                    + 0.03 * lost_age
                )
                cost[lost_idx, col_idx] = max(
                    0.0,
                    base_cost + self.low_point_cost_adjustment(track, detection),
                )

        row_indices, col_indices = linear_sum_assignment(cost)

        matched = []
        matched_lost_indices = set()
        matched_detection_indices = set()

        for lost_idx, col_idx in zip(row_indices, col_indices):
            if cost[lost_idx, col_idx] >= 1e6:
                continue

            detection_idx = int(detection_indices[col_idx])
            matched.append((int(lost_idx), detection_idx))
            matched_lost_indices.add(int(lost_idx))
            matched_detection_indices.add(detection_idx)

        remaining_unmatched_detections = [
            detection_idx for detection_idx in detection_indices if detection_idx not in matched_detection_indices
        ]

        return matched, remaining_unmatched_detections, matched_lost_indices
