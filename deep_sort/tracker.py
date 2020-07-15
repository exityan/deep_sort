# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import logging
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


logger = logging.getLogger(__name__)


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        Returns
        -------
        List[deep_sort.track.Track]
            Tracks for input detections

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        tracks = [None] * len(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            tracks[detection_idx] = self.tracks[track_idx]
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            track = self._initiate_track(detections[detection_idx])
            tracks[detection_idx] = track
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

        return tracks

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            logger.debug(f"[_match] targets: {targets}")
            cost_matrix = self.metric.distance(features, targets)
            logger.debug(f"[_match] cost_matrix: {cost_matrix}")
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            logger.debug(f"[_match] gate cost_matrix: {cost_matrix}")

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        logger.debug(f"[_match] Associate confirmed tracks using appearance features.")

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        logger.debug(f"[_match] Associate remaining tracks together with unconfirmed tracks using IOU")

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        logger.debug(f"[_match] iou_track_candidates: {iou_track_candidates}")

        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        logger.debug(f"[_match] unmatched_tracks_a: {unmatched_tracks_a}")
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        logger.debug(f"[_match] matches_b: {matches_b}")
        logger.debug(f"[_match] unmatched_tracks_b: {unmatched_tracks_b}")
        logger.debug(f"[_match] unmatched_detections: {unmatched_detections}")

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        logger.debug(f"[_match] matches: {matches}")
        logger.debug(f"[_match] unmatched_tracks: {unmatched_tracks}")
        logger.debug(f"[_match] unmatched_detections: {unmatched_detections}")

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """
        Returns
        -------
        Track
            initiated track

        """
        mean, covariance = self.kf.initiate(detection.to_xyah())
        track_id = self._next_id
        self._next_id += 1
        track = Track(mean, covariance, track_id, self.n_init, self.max_age,
                      detection.feature)
        self.tracks.append(track)
        return track
