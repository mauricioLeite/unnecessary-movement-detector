import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

import movement_detector


class ProgressTests(unittest.TestCase):
    def test_noninteractive_output_only_reports_completion(self):
        output = io.StringIO()

        with redirect_stdout(output):
            update = movement_detector.make_progress("pose", 10)
            update(0)
            update(9, force=True)

        self.assertEqual(output.getvalue(), "  pose: 10/10\n")


class SmoothSignalTests(unittest.TestCase):
    def test_short_signals_keep_their_length(self):
        for length in range(1, 5):
            with self.subTest(length=length):
                times = np.arange(length, dtype=float)
                values = np.arange(length, dtype=float)
                smoothed, dt = movement_detector.smooth_signal(times, values)

                self.assertEqual(len(smoothed), length)
                self.assertGreater(dt, 0)


class PeakCounterTests(unittest.TestCase):
    def test_counts_distinct_peaks(self):
        signal = np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.9, 0.0])

        count, peaks = movement_detector.count_reps_from_peaks(
            signal,
            dt=0.1,
            prominence=0.1,
            min_gap_s=0.2,
            depth_frac=0.5,
        )

        self.assertEqual(count, 2)
        np.testing.assert_array_equal(peaks, [2, 5])

    def test_short_signal_returns_integer_indices(self):
        count, peaks = movement_detector.count_reps_from_peaks(
            np.array([0.0, 0.5, 0.0]),
            dt=0.1,
            prominence=0.1,
            min_gap_s=1.0,
            depth_frac=0.5,
        )

        self.assertEqual(count, 0)
        self.assertEqual(peaks.dtype.kind, "i")


class StateMachineTests(unittest.TestCase):
    def count(self, signal, min_gap=0.0):
        values = np.array(signal, dtype=float)
        times = np.arange(len(values), dtype=float) * 0.1
        return movement_detector.count_reps_state_machine(
            times,
            values,
            enter_frac=0.6,
            leave_frac=0.4,
            stand_frac=0.25,
            min_gap_s=min_gap,
        )

    def test_counts_completed_cycle_and_rejects_aborted_descent(self):
        signal = (
            [0.0] * 10
            + [0.35]
            + [0.0] * 10
            + [0.4] * 2
            + [0.8] * 4
            + [0.3] * 2
            + [0.0] * 10
        )

        count, rep_indices, phases = self.count(signal)

        self.assertEqual(count, 1)
        self.assertEqual(len(rep_indices), 1)
        self.assertEqual(len(phases), len(signal))

    def test_redip_during_ascent_stays_in_the_same_cycle(self):
        signal = (
            [0.0] * 10
            + [0.4] * 2
            + [0.8] * 3
            + [0.3] * 2
            + [0.8] * 3
            + [0.3] * 2
            + [0.0] * 10
        )

        count, _, _ = self.count(signal)

        self.assertEqual(count, 1)

    def test_incomplete_cycle_is_not_counted(self):
        signal = [0.0] * 10 + [0.4] * 2 + [0.8] * 4

        count, _, _ = self.count(signal)

        self.assertEqual(count, 0)

    def test_minimum_gap_rejects_a_second_close_cycle(self):
        cycle = [0.4] * 2 + [0.8] * 4 + [0.3] * 2 + [0.0] * 3
        signal = [0.0] * 10 + cycle + cycle + [0.0] * 10

        count, _, _ = self.count(signal, min_gap=10.0)

        self.assertEqual(count, 1)


class ArgumentTests(unittest.TestCase):
    def test_defaults(self):
        args = movement_detector.parse_args(["video.mp4"])

        self.assertEqual(args.frame_step, 1)
        self.assertEqual(args.model_complexity, 1)
        self.assertEqual(args.min_gap, 1.0)
        self.assertFalse(args.no_video)

    def test_invalid_values_are_rejected(self):
        invalid_arguments = [
            ["--frame-step", "0"],
            ["--min-gap", "-1"],
            ["--prominence", "-0.1"],
            ["--depth-frac", "1.1"],
            ["--stand-frac", "-0.1"],
            ["--stand-frac", "0.5", "--leave-frac", "0.4"],
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        movement_detector.parse_args(["video.mp4", *arguments])

    def test_version_is_reported_without_a_video(self):
        output = io.StringIO()

        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as exit_status:
                movement_detector.parse_args(["--version"])

        self.assertEqual(exit_status.exception.code, 0)
        self.assertIn(movement_detector.__version__, output.getvalue())


if __name__ == "__main__":
    unittest.main()
