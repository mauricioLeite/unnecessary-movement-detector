"""Count burpees in a video and optionally render an annotated copy."""

import argparse
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks, savgol_filter

__version__ = "0.1.0"

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Configuration
OUTPUT_HEIGHT = 0
SMOOTH_SECONDS = 0.4
FLASH_SECONDS = 0.5
PROGRESS_BAR_WIDTH = 30
PROGRESS_MIN_INTERVAL = 0.2

# OpenCV colors use BGR order.
OVERLAY_BG = (18, 18, 18)
OVERLAY_TEXT = (245, 245, 245)
OVERLAY_MUTED = (190, 190, 190)
OVERLAY_ACCENT = (80, 210, 80)


def make_progress(label, total):
    """Return a throttled terminal progress updater."""
    start = last = time.time()
    interactive = sys.stdout.isatty()

    def update(i, force=False):
        nonlocal last
        if not interactive:
            if force:
                print(f"  {label}: {max(0, i + 1)}/{total}")
            return
        now = time.time()
        done = force or (total > 0 and i + 1 >= total)
        if not done and now - last < PROGRESS_MIN_INTERVAL:
            return
        last = now
        frac = min(1.0, (i + 1) / total) if total else 1.0
        filled = int(PROGRESS_BAR_WIDTH * frac)
        bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
        elapsed = now - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
        eta = max(0, total - i - 1) / rate if rate > 0 else 0.0
        print(f"\r  {label} [{bar}] {frac * 100:5.1f}%  {i + 1}/{total}  {rate:5.1f} fps  ETA {eta:5.0f}s   ", end="\n" if done else "", flush=True,)

    return update


def analyze_video(video_path, frame_step=1, model_complexity=1):
    """Extract pose landmarks and body-height samples from a video.

    Pose results are reused on frames skipped by ``frame_step``.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total} frames @ {fps:.2f} fps (~{total / fps:.1f}s), {width}x{height}")

    landmarks = []
    sample_frames = []
    body_heights = []

    progress = make_progress("pose", total)
    with mp_pose.Pose(model_complexity=model_complexity, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_index = 0
        last_landmarks = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                last_landmarks = result.pose_landmarks or None
                if last_landmarks is not None:
                    # MediaPipe y increases downward; ignore uncertain landmarks.
                    visible_y = [
                        landmark.y
                        for landmark in last_landmarks.landmark
                        if landmark.visibility > 0.5
                    ]
                    if visible_y:
                        sample_frames.append(frame_index)
                        body_heights.append(float(np.mean(visible_y)))
            # Keep the last pose so skipped frames do not flicker when rendered.
            landmarks.append(last_landmarks)
            progress(frame_index)
            frame_index += 1
    progress(frame_index - 1, force=True)
    cap.release()
    return fps, (width, height), landmarks, sample_frames, body_heights


def smooth_signal(times, values):
    """Smooth body height over a fixed duration and return the sample period."""
    dt = np.median(np.diff(times)) if len(times) > 1 else 1 / 30
    win = int(round(SMOOTH_SECONDS / dt))
    # Savitzky-Golay windows must be odd and longer than the cubic polynomial.
    if win % 2 == 0:
        win += 1
    win = max(win, 5)
    if win > len(values):
        win = (len(values) - 1) | 1
    return savgol_filter(values, win, 3) if win >= 5 else np.asarray(values), dt


def count_reps_from_peaks(smoothed, dt, prominence, min_gap_s, depth_frac):
    """Count repetitions from peaks in the body-height signal."""
    if len(smoothed) < 5:
        return 0, np.array([], dtype=int)

    # Percentiles prevent an outlier from setting the required depth.
    lo, hi = np.percentile(smoothed, 5), np.percentile(smoothed, 95)
    height_thr = lo + depth_frac * (hi - lo)

    # Larger y values put the body lower in the frame, so bottoms are peaks.
    min_distance = max(1, int(min_gap_s / dt))
    peaks, _ = find_peaks(smoothed, prominence=prominence, distance=min_distance, height=height_thr)
    return len(peaks), peaks


STANDING, DESCENDING, BOTTOM, ASCENDING = 0, 1, 2, 3


def count_reps_state_machine(times, smoothed, enter_frac, leave_frac, stand_frac, min_gap_s):
    """Count complete movement cycles with hysteresis at the bottom phase.

    A repetition is recorded after returning to standing.
    """
    n = len(smoothed)
    if n < 5:
        return 0, np.array([], dtype=int), np.zeros(n, dtype=int)

    lo, hi = np.percentile(smoothed, 5), np.percentile(smoothed, 95)
    span = hi - lo
    enter_threshold = lo + enter_frac * span
    leave_threshold = lo + leave_frac * span
    stand_threshold = lo + stand_frac * span

    phases = np.empty(n, dtype=int)
    state = STANDING
    rep_indices = []
    last_rep_time = -np.inf

    for index, sample in enumerate(smoothed):
        if state == STANDING:
            if sample > leave_threshold:
                state = DESCENDING
        elif state == DESCENDING:
            if sample >= enter_threshold:
                state = BOTTOM
            elif sample < stand_threshold:
                state = STANDING
        elif state == BOTTOM:
            if sample < leave_threshold:
                state = ASCENDING
        elif state == ASCENDING:
            if sample >= enter_threshold:
                state = BOTTOM
            elif sample <= stand_threshold:
                state = STANDING
                if times[index] - last_rep_time >= min_gap_s:
                    rep_indices.append(index)
                    last_rep_time = times[index]
        phases[index] = state

    return len(rep_indices), np.array(rep_indices, dtype=int), phases


def save_plot(times, raw, smoothed, peaks, state_rep_indices, state_phases, thresholds, out_path,):
    """Save a plot comparing peak and state-machine results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    enter_threshold, leave_threshold, stand_threshold = thresholds

    plt.figure(figsize=(14, 5))

    in_bottom = state_phases == BOTTOM
    plt.fill_between(times, 0, 1, where=in_bottom, transform=plt.gca().get_xaxis_transform(), color="#ffddaa", alpha=0.4, step="pre", label="state: bottom",)

    plt.plot(times, raw, color="#bbb", lw=1, label="raw centroid height")
    plt.plot(times, smoothed, color="#1f77b4", lw=2, label="smoothed")

    plt.axhline(enter_threshold, color="#d62728", lw=1, ls="--", alpha=0.7, label="enter (bottom)",)
    plt.axhline(leave_threshold, color="#ff7f0e", lw=1, ls="--", alpha=0.7, label="leave (bottom)",)
    plt.axhline(stand_threshold, color="#2ca02c", lw=1, ls="--", alpha=0.7, label="standing", )

    plt.plot(times[peaks], smoothed[peaks], "rv", ms=12, label=f"find peaks reps ({len(peaks)})", )
    plt.plot(times[state_rep_indices], smoothed[state_rep_indices], "go", ms=10, label=f"state machine reps ({len(state_rep_indices)})", )

    plt.gca().invert_yaxis()
    plt.xlabel("time (s)")
    plt.ylabel("body height (normalized, inverted)")
    plt.title( f"find peaks: {len(peaks)}   |   state machine: {len(state_rep_indices)}" )
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"Saved verification plot -> {out_path}")


def render_video(video_path, out_path, landmarks, rep_frames, fps, source_size):
    """Render cached landmarks and repetition feedback over the source video."""
    source_width, source_height = source_size
    if OUTPUT_HEIGHT and OUTPUT_HEIGHT != source_height:
        scale = OUTPUT_HEIGHT / source_height
        output_width = int(round(source_width * scale))
        output_height = OUTPUT_HEIGHT
    else:
        output_width, output_height = source_width, source_height
    flash_frames = int(FLASH_SECONDS * fps)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not reopen video for rendering: {video_path}")

    # The Make pipeline transcodes this broadly available writer codec to H.264.
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))
    if not writer.isOpened():
        cap.release()
        sys.exit(f"Could not create annotated video: {out_path}")

    ui = max(0.65, min(2.0, output_height / 720.0))
    margin = max(10, int(round(20 * ui)))
    pad_x = max(10, int(round(16 * ui)))
    pad_y = max(8, int(round(12 * ui)))
    gap = max(8, int(round(12 * ui)))
    accent_w = max(4, int(round(5 * ui)))
    label_scale = 0.55 * ui
    count_scale = 1.35 * ui
    chip_scale = 0.62 * ui
    label_thick = max(1, int(round(1.5 * ui)))
    count_thick = max(2, int(round(2.5 * ui)))
    chip_thick = max(1, int(round(2 * ui)))

    label = "BURPEES"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thick)

    def dim_panel(x0, y0, x1, y1, color=OVERLAY_BG, opacity=0.72):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(output_width, x1), min(output_height, y1)
        if x1 <= x0 or y1 <= y0:
            return
        roi = frame[y0:y1, x0:x1]
        tint = np.full_like(roi, color)
        cv2.addWeighted(tint, opacity, roi, 1.0 - opacity, 0, roi)

    progress = make_progress("render", len(landmarks))
    count = 0
    last_rep_frame = None
    frame_index = -1
    for frame_index in range(len(landmarks)):
        ok, frame = cap.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != (output_width, output_height):
            # MediaPipe landmarks are normalized and remain valid after resizing.
            frame = cv2.resize(frame, (output_width, output_height))

        if landmarks[frame_index] is not None:
            mp_draw.draw_landmarks(frame, landmarks[frame_index], mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(), )

        while count < len(rep_frames) and rep_frames[count] <= frame_index:
            last_rep_frame = rep_frames[count]
            count += 1

        count_text = str(count)
        count_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, count_scale, count_thick)
        content_h = max(label_size[1], count_size[1])
        card_x0, card_y0 = margin, margin
        card_w = accent_w + pad_x * 3 + label_size[0] + count_size[0]
        card_h = content_h + pad_y * 2
        card_x1, card_y1 = card_x0 + card_w, card_y0 + card_h
        dim_panel(card_x0, card_y0, card_x1, card_y1)
        cv2.rectangle(frame, (card_x0, card_y0), (card_x0 + accent_w, card_y1), OVERLAY_ACCENT, -1)

        baseline = card_y0 + pad_y + content_h
        label_x = card_x0 + accent_w + pad_x
        cv2.putText(frame, label, (label_x, baseline), cv2.FONT_HERSHEY_SIMPLEX, label_scale, OVERLAY_MUTED, label_thick, cv2.LINE_AA)
        count_x = label_x + label_size[0] + pad_x
        cv2.putText(frame, count_text, (count_x, baseline), cv2.FONT_HERSHEY_SIMPLEX, count_scale, OVERLAY_TEXT,
                    count_thick, cv2.LINE_AA)

        if last_rep_frame is not None and flash_frames > 0:
            rep_age = frame_index - last_rep_frame
            if 0 <= rep_age < flash_frames:
                progress_through_flash = rep_age / max(1, flash_frames - 1)
                fade = 1.0 if progress_through_flash < 0.45 else max(0.0, (1.0 - progress_through_flash) / 0.55)
                chip_text = "+1 REP"
                chip_size, _ = cv2.getTextSize(
                    chip_text, cv2.FONT_HERSHEY_SIMPLEX, chip_scale, chip_thick)
                chip_w = chip_size[0] + pad_x * 2
                chip_h = max(chip_size[1] + pad_y * 2, card_h)
                chip_x0, chip_y0 = card_x1 + gap, card_y0
                if chip_x0 + chip_w > output_width - margin:
                    chip_x0, chip_y0 = card_x0, card_y1 + gap
                chip_x1, chip_y1 = chip_x0 + chip_w, chip_y0 + chip_h

                overlay = frame.copy()
                cv2.rectangle(overlay, (chip_x0, chip_y0), (chip_x1, chip_y1), OVERLAY_ACCENT, -1)
                chip_baseline = chip_y0 + (chip_h + chip_size[1]) // 2
                cv2.putText(overlay, chip_text, (chip_x0 + pad_x, chip_baseline), cv2.FONT_HERSHEY_SIMPLEX, chip_scale, (15, 15, 15), chip_thick, cv2.LINE_AA)
                cv2.addWeighted(overlay, fade, frame, 1.0 - fade, 0, frame)

        writer.write(frame)
        progress(frame_index)
    progress(frame_index, force=True)
    cap.release()
    writer.release()
    print(f"Saved annotated video -> {out_path}")


def parse_args(argv=None):
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Count burpees with find_peaks and a state machine, then optionally "
            "render an annotated video"
        )
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("video", help="path to input video")
    parser.add_argument(
        "--out", default="annotated.mp4", help="annotated video output"
    )
    parser.add_argument(
        "--plot", default="signal.png", help="verification plot output"
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=0.04,
        help="[find_peaks] minimum peak prominence in normalized units",
    )
    parser.add_argument(
        "--depth-frac",
        type=float,
        default=0.5,
        help="[find_peaks] required depth within the observed range [0-1]",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=1.0,
        help="minimum seconds between repetitions for both methods",
    )
    parser.add_argument(
        "--enter-frac",
        type=float,
        default=0.6,
        help="[state machine] fraction of range that enters BOTTOM",
    )
    parser.add_argument(
        "--leave-frac",
        type=float,
        default=0.4,
        help="[state machine] fraction of range that leaves BOTTOM",
    )
    parser.add_argument(
        "--stand-frac",
        type=float,
        default=0.25,
        help="[state machine] fraction of range considered STANDING",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="process every Nth frame for pose estimation",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe pose model: 0=lite, 1=full, 2=heavy",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="skip annotated video rendering",
    )
    args = parser.parse_args(argv)

    if args.frame_step < 1:
        parser.error("--frame-step must be at least 1")
    if args.min_gap < 0:
        parser.error("--min-gap must be non-negative")
    if args.prominence < 0:
        parser.error("--prominence must be non-negative")
    if not 0 <= args.depth_frac <= 1:
        parser.error("--depth-frac must be between 0 and 1")

    fractions = (args.stand_frac, args.leave_frac, args.enter_frac)
    if not all(0 <= value <= 1 for value in fractions):
        parser.error("state-machine fractions must be between 0 and 1")
    if not args.stand_frac < args.leave_frac < args.enter_frac:
        parser.error("expected --stand-frac < --leave-frac < --enter-frac")

    return args


def main(argv=None):
    args = parse_args(argv)

    fps, size, landmarks, sample_frames, body_heights = analyze_video(args.video, args.frame_step, args.model_complexity )
    if not body_heights:
        sys.exit("No pose detected in any frame - check camera framing.")

    times = np.array(sample_frames) / fps
    body_heights = np.array(body_heights)
    smoothed, dt = smooth_signal(times, body_heights)

    peak_count, peaks = count_reps_from_peaks(smoothed, dt, args.prominence, args.min_gap, args.depth_frac )
    state_count, state_rep_indices, state_phases = count_reps_state_machine(
        times,
        smoothed,
        args.enter_frac,
        args.leave_frac,
        args.stand_frac,
        args.min_gap,
    )

    peak_rep_frames = sorted(sample_frames[index] for index in peaks)
    state_rep_frames = sorted(sample_frames[index] for index in state_rep_indices)

    print("\n" + "=" * 50)
    print(f"  find_peaks    : {peak_count} reps")
    print(f"  state machine : {state_count} reps")
    print("=" * 50)
    if peak_rep_frames:
        print("find_peaks timestamps:    " + ", ".join(f"{frame / fps:.1f}s" for frame in peak_rep_frames))
    if state_rep_frames:
        print("state machine timestamps: " + ", ".join(f"{frame / fps:.1f}s" for frame in state_rep_frames))

    lo, hi = np.percentile(smoothed, 5), np.percentile(smoothed, 95)
    span = hi - lo
    thresholds = (lo + args.enter_frac * span, lo + args.leave_frac * span, lo + args.stand_frac * span)
    save_plot(
        times,
        body_heights,
        smoothed,
        peaks,
        state_rep_indices,
        state_phases,
        thresholds,
        args.plot,
    )
    if not args.no_video:
        render_video( args.video, args.out, landmarks, state_rep_frames, fps, size )


if __name__ == "__main__":
    main()
