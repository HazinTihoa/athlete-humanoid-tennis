import unittest

import numpy as np

from athlete.goal_cond_tracking.tennis_planner import (
    MotionReachTarget,
    MotionTrajectoryMatch,
    StrokeSide,
    closest_trajectory_point,
    filter_motion_matches,
    first_forward_plane_crossing,
    rank_motion_matches,
    retain_motion_match,
)


class TennisPlannerTest(unittest.TestCase):
    def test_forward_plane_crossing_classifies_forehand(self) -> None:
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[2.0, -0.5, 1.0], [0.5, -0.6, 0.9], [-1.0, -0.7, 0.8]])

        crossing = first_forward_plane_crossing(
            times,
            positions,
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            side_deadband=0.1,
        )

        self.assertIsNotNone(crossing)
        assert crossing is not None
        self.assertEqual(crossing.side, StrokeSide.FOREHAND)
        self.assertAlmostEqual(crossing.position_world[0], 0.0)
        self.assertLess(crossing.lateral_position, -0.1)

    def test_reverse_crossing_is_ignored(self) -> None:
        crossing = first_forward_plane_crossing(
            np.array([0.0, 1.0]),
            np.array([[-1.0, 0.5, 1.0], [1.0, 0.5, 1.0]]),
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        self.assertIsNone(crossing)

    def test_closest_point_uses_segment_projection_and_interpolated_time(self) -> None:
        distance, point, arrival_time, segment = closest_trajectory_point(
            np.array([0.0, 2.0]),
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([0.75, 0.2, 0.0]),
        )

        self.assertAlmostEqual(distance, 0.2)
        np.testing.assert_allclose(point, [0.75, 0.0, 0.0])
        self.assertAlmostEqual(arrival_time, 0.75)
        self.assertEqual(segment, 0)

    def test_ranking_filters_side_distance_and_minimum_strike_time(self) -> None:
        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array(
            [[3.0, -0.5, 1.0], [2.0, -0.5, 1.0], [1.0, -0.5, 1.0], [0.0, -0.5, 1.0]]
        )
        motions = [
            MotionReachTarget(
                0, StrokeSide.FOREHAND, np.array([1.1, -0.5, 1.0]), 1.0, 0.5
            ),
            MotionReachTarget(
                1, StrokeSide.BACKHAND, np.array([1.0, -0.5, 1.0]), 1.0, 0.5
            ),
            MotionReachTarget(
                2, StrokeSide.FOREHAND, np.array([1.0, -0.8, 1.0]), 1.0, 0.5
            ),
            MotionReachTarget(
                3, StrokeSide.FOREHAND, np.array([2.0, -0.5, 1.0]), 1.5, 1.1
            ),
        ]

        ranked = rank_motion_matches(
            times,
            positions,
            motions,
            side=StrokeSide.FOREHAND,
            max_distance=0.2,
        )

        self.assertEqual([match.motion_index for match in ranked], [0])
        self.assertAlmostEqual(ranked[0].distance, 0.0)
        self.assertAlmostEqual(ranked[0].arrival_time, 1.9)
        self.assertAlmostEqual(ranked[0].contact_time, 1.0)
        self.assertAlmostEqual(ranked[0].wait_time, 0.9)
        self.assertAlmostEqual(ranked[0].time_slack, 1.4)

    def test_ball_arrival_inside_training_range_compresses_motion(self) -> None:
        times = np.array([0.0, 4.0])
        positions = np.array([[0.0, 0.0, 1.0], [4.0, 0.0, 1.0]])
        motions = [
            MotionReachTarget(
                0,
                StrokeSide.FOREHAND,
                np.array([2.082, 0.0, 1.0]),
                3.360,
                1.680,
            )
        ]

        ranked = rank_motion_matches(
            times,
            positions,
            motions,
            side=StrokeSide.FOREHAND,
            reaction_margin=0.3,
        )

        self.assertEqual(len(ranked), 1)
        self.assertAlmostEqual(ranked[0].arrival_time, 2.082)
        self.assertAlmostEqual(ranked[0].contact_time, 2.082)
        self.assertAlmostEqual(ranked[0].wait_time, 0.0)
        self.assertAlmostEqual(ranked[0].time_slack, 0.402)

    def test_ball_arrival_below_minimum_strike_time_is_rejected(self) -> None:
        times = np.array([0.0, 2.0])
        positions = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
        motions = [
            MotionReachTarget(
                0,
                StrokeSide.FOREHAND,
                np.array([1.5, 0.0, 1.0]),
                3.360,
                1.680,
            )
        ]

        ranked = rank_motion_matches(
            times,
            positions,
            motions,
            side=StrokeSide.FOREHAND,
        )

        self.assertEqual(ranked, [])

    def test_center_crossing_can_rank_both_sides(self) -> None:
        times = np.array([0.0, 1.0])
        positions = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        motions = [
            MotionReachTarget(
                0, StrokeSide.FOREHAND, np.array([0.5, -0.05, 1.0]), 0.5, 0.1
            ),
            MotionReachTarget(
                1, StrokeSide.BACKHAND, np.array([0.5, 0.02, 1.0]), 0.5, 0.1
            ),
        ]

        ranked = rank_motion_matches(
            times,
            positions,
            motions,
            side=None,
            max_distance=0.2,
        )

        self.assertEqual([match.motion_index for match in ranked], [1, 0])

    def test_prepared_motion_survives_initial_reaction_margin(self) -> None:
        match = MotionTrajectoryMatch(
            motion_index=7,
            side=StrokeSide.FOREHAND,
            distance=0.1,
            target_world=np.array([1.0, -0.5, 1.0]),
            arrival_time=1.2,
            nominal_strike_time=1.0,
            minimum_strike_time=1.0,
            contact_time=1.0,
            wait_time=0.2,
            time_slack=0.2,
            segment_index=0,
        )

        self.assertEqual(
            filter_motion_matches([match], max_distance=0.2, reaction_margin=0.3),
            [],
        )
        self.assertIs(
            retain_motion_match([match], 7, max_distance=0.2),
            match,
        )

    def test_prepared_motion_is_dropped_when_too_fast_or_out_of_range(self) -> None:
        def match(distance: float, time_slack: float) -> MotionTrajectoryMatch:
            return MotionTrajectoryMatch(
                motion_index=3,
                side=StrokeSide.BACKHAND,
                distance=distance,
                target_world=np.zeros(3),
                arrival_time=1.0 + time_slack,
                nominal_strike_time=1.0,
                minimum_strike_time=1.0,
                contact_time=min(1.0 + time_slack, 1.0),
                wait_time=max(time_slack, 0.0),
                time_slack=time_slack,
                segment_index=0,
            )

        self.assertIsNone(
            retain_motion_match([match(0.21, 0.1)], 3, max_distance=0.2)
        )
        self.assertIsNone(
            retain_motion_match([match(0.1, -0.04)], 3, max_distance=0.2)
        )


if __name__ == "__main__":
    unittest.main()
