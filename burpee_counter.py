"""Count burpees in a video and render an annotated version — one pass.

This version COMPARES two counting methods on the same signal:
  A. find_peaks   — the original approach: treat each dip as an isolated peak.
  B. state machine — a rep is a full cycle
       standing -> descending -> bottom -> ascending -> standing,
     counted only on completion, with hysteresis (separate enter/leave
     thresholds for "bottom") so flat bottoms and brief pose noise don't
     create phantom transitions the way isolated peak-picking can.

=============================================================================
THE IDEA (no machine-learning training required)
=============================================================================
A burpee is a periodic motion: the body drops to the ground (plank) and rises
to a standing jump, once per repetition. If we turn "how high is the body in
the frame" into one number per frame, that number oscillates with exactly ONE
dip per burpee, so counting burpees = counting the dips.

Pipeline:
  1. POSE     Run pretrained MediaPipe Pose on every frame -> 33 body
              landmarks. (This is the only "AI" component; it isn't trained
              by us.)
  2. SIGNAL   Collapse the landmarks to one number per frame: the average
              vertical position of the visible joints (the body's centroid
              height). MediaPipe y is 0 at the top, 1 at the bottom, so a
              LARGER value means the body is LOWER (closer to the ground).
  3. COUNT    Smooth the signal once, then count reps BOTH ways (find_peaks
              and the phase state machine) so they can be compared directly.
  4. OUTPUTS  Print both counts, save a comparison plot, and render an
              annotated video (skeleton + live counter + rep feedback) driven
              by the state machine's reps.

We run pose estimation ONCE and reuse the cached landmarks for both the count
and the annotated video.
=============================================================================
"""

import argparse
import sys
import time

import cv2                                    # read/write video, draw overlays
import numpy as np                            # numeric arrays / math
import mediapipe as mp                        # pretrained pose estimator
from scipy.signal import find_peaks, savgol_filter

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils          # draws the skeleton
mp_styles = mp.solutions.drawing_styles

# --- Fixed settings (edit here; intentionally NOT command-line flags) -------
OUTPUT_HEIGHT = 0         # output video height in px; 0 = keep source resolution
SMOOTH_SECONDS = 0.4      # smoothing window as a real duration (fps-independent)
FLASH_SECONDS = 0.5       # how long the "+1 REP" chip stays after each rep
PROGRESS_BAR_WIDTH = 30   # characters wide, for the terminal progress bar
PROGRESS_MIN_INTERVAL = 0.2   # seconds between redraws (throttles terminal spam)

# Overlay palette (OpenCV uses BGR, not RGB).
OVERLAY_BG = (18, 18, 18)
OVERLAY_TEXT = (245, 245, 245)
OVERLAY_MUTED = (190, 190, 190)
OVERLAY_ACCENT = (80, 210, 80)


def make_progress(label, total):
    """Return update(i) which redraws an in-place [####----] bar with rate/ETA.

    Throttled by wall-clock time (not frame count) so it looks smooth whether
    the loop is doing full pose inference or skipping frames via --frame-step.
    """
    start = last = time.time()

    def update(i, force=False):
        nonlocal last
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
        print(f"\r  {label} [{bar}] {frac * 100:5.1f}%  {i + 1}/{total}" f"  {rate:5.1f} fps  ETA {eta:5.0f}s   ", end="\n" if done else "", flush=True)

    return update


def analyze(video_path, frame_step=1, model_complexity=1):
    """Single pose pass over the video.

    frame_step runs pose on every Nth frame only (1 = every frame); skipped
    frames reuse the last processed landmarks so the annotated video doesn't
    flicker. model_complexity is passed straight through to MediaPipe
    (0=lite/fast, 1=full, 2=heavy).

    Returns:
      fps        : frames per second
      size       : (width, height) of the source frames
      landmarks  : list with one entry per frame — the pose landmarks, or None
      valid_idx  : frame indices that produced a usable body-height value
      values     : the body-height value for each valid frame (larger = lower)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total} frames @ {fps:.2f} fps (~{total / fps:.1f}s), {w}x{h}")

    landmarks = []
    valid_idx, values = [], []

    # Build the pose model once and reuse it for every frame (creating it is
    # expensive). model_complexity 0/1/2 trades speed for accuracy.
    progress = make_progress("pose", total)
    with mp_pose.Pose(model_complexity=model_complexity,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        i = 0
        last_lm = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % frame_step == 0:
                # OpenCV gives BGR; MediaPipe wants RGB.
                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                last_lm = res.pose_landmarks if res.pose_landmarks else None
                if last_lm is not None:
                    # Body-height = mean y of confidently-visible joints. Ignoring
                    # low-visibility joints keeps a half-guessed limb from skewing it.
                    ys = [lm.y for lm in last_lm.landmark if lm.visibility > 0.5]
                    if ys:
                        valid_idx.append(i)
                        values.append(float(np.mean(ys)))
            landmarks.append(last_lm)   # carry forward on skipped frames
            progress(i)
            i += 1
    progress(i - 1, force=True)   # ensure a clean final newline even if `total` was off
    cap.release()
    return fps, (w, h), landmarks, valid_idx, values


def smooth_signal(times, values):
    """Savitzky-Golay smoothing shared by both counting methods.

    Fits a small polynomial over a sliding window, removing jitter while
    preserving dip shape/depth. Window is sized from a real duration so it's
    the same regardless of fps. It must be odd and >= 5.
    """
    dt = np.median(np.diff(times)) if len(times) > 1 else 1 / 30
    win = int(round(SMOOTH_SECONDS / dt))
    if win % 2 == 0:
        win += 1
    win = max(win, 5)
    if win > len(values):
        win = (len(values) - 1) | 1
    return savgol_filter(values, win, 3) if win >= 5 else np.asarray(values), dt


def count_reps(times, values, smooth, dt, prominence, min_gap_s, depth_frac):
    """Count the ground-contact dips in the body-height signal (find_peaks).

    Returns (count, peak_indices). peak_indices index into times/values at
    each detected rep.
    """
    if len(values) < 5:
        return 0, np.array([])

    # --- Depth threshold ---
    # Only count dips that truly reach toward the ground; this rejects standing
    # sways and start/end resting positions. Percentiles (not min/max) keep one
    # outlier frame from stretching the range.
    lo, hi = np.percentile(smooth, 5), np.percentile(smooth, 95)
    height_thr = lo + depth_frac * (hi - lo)

    # --- Peak detection ---
    # Bigger value = lower body = ground, so a burpee bottom is a MAXIMUM here.
    #   height     -> deep enough (near the ground)
    #   prominence -> stands out from the surrounding signal
    #   distance   -> at least min_gap_s apart (no double counting)
    min_distance = max(1, int(min_gap_s / dt))
    peaks, _ = find_peaks(smooth, prominence=prominence,
                          distance=min_distance, height=height_thr)
    return len(peaks), peaks


# --- State-machine phase labels (numeric, for cheap storage/plotting) -------
STANDING, DESCENDING, BOTTOM, ASCENDING = 0, 1, 2, 3
PHASE_NAMES = {STANDING: "standing", DESCENDING: "descending",
               BOTTOM: "bottom", ASCENDING: "ascending"}


def count_reps_state_machine(times, smooth, enter_frac, leave_frac,
                             stand_frac, min_gap_s):
    """Count reps by walking a phase/state machine over the smoothed signal.

    Models a rep as standing -> descending -> bottom -> ascending -> standing
    and counts only when the athlete completes the cycle back to standing.
    Hysteresis (enter_frac > leave_frac) means entering and leaving the
    "bottom" state use different thresholds, so a flat/noisy bottom doesn't
    flicker in and out and a single dip isn't double-counted.

    Returns (count, rep_indices, phase). rep_indices index into times/smooth
    at the sample where each rep completed (used to place it in time).
    phase[i] is the state machine's phase at sample i, for plotting.
    """
    n = len(smooth)
    if n < 5:
        return 0, np.array([], dtype=int), np.zeros(n, dtype=int)

    # Bigger value = lower body = ground. Fractions are of the same robust
    # [5th, 95th] percentile range used by the find_peaks method, so the two
    # methods are tuned on comparable scales.
    lo, hi = np.percentile(smooth, 5), np.percentile(smooth, 95)
    span = hi - lo
    enter_thr = lo + enter_frac * span   # cross UP through this -> BOTTOM
    leave_thr = lo + leave_frac * span   # drop below this -> ASCENDING
    stand_thr = lo + stand_frac * span   # back below this -> STANDING (counts)

    phase = np.empty(n, dtype=int)
    state = STANDING
    rep_indices = []
    last_rep_time = -np.inf

    for i in range(n):
        s = smooth[i]
        if state == STANDING:
            if s > leave_thr:
                state = DESCENDING
        elif state == DESCENDING:
            if s >= enter_thr:
                state = BOTTOM
            elif s < stand_thr:
                state = STANDING          # aborted dip, never reached bottom
        elif state == BOTTOM:
            if s < leave_thr:
                state = ASCENDING          # flat bottoms just stay in BOTTOM
        elif state == ASCENDING:
            if s >= enter_thr:
                state = BOTTOM             # dipped again mid-rise, same rep
            elif s <= stand_thr:
                state = STANDING
                if times[i] - last_rep_time >= min_gap_s:
                    rep_indices.append(i)
                    last_rep_time = times[i]
        phase[i] = state

    return len(rep_indices), np.array(rep_indices, dtype=int), phase


def save_plot(times, raw, smooth, peaks, sm_reps, sm_phase, thresholds, out_path):
    """Save the body-height wave comparing find_peaks vs the state machine.

    peaks     -> find_peaks rep indices (into times/smooth)
    sm_reps   -> state-machine rep indices (into times/smooth)
    sm_phase  -> state-machine phase per sample (for BOTTOM shading)
    thresholds -> (enter_thr, leave_thr, stand_thr) hysteresis lines
    """
    import matplotlib
    matplotlib.use("Agg")                     # headless backend (no display)
    import matplotlib.pyplot as plt

    enter_thr, leave_thr, stand_thr = thresholds

    plt.figure(figsize=(14, 5))

    # Shade the state machine's BOTTOM phase so hysteresis behavior is visible.
    in_bottom = sm_phase == BOTTOM
    plt.fill_between(times, 0, 1, where=in_bottom, transform=plt.gca().get_xaxis_transform(), color="#ffddaa", alpha=0.4, step="pre", label="state: bottom")

    plt.plot(times, raw, color="#bbb", lw=1, label="raw centroid height")
    plt.plot(times, smooth, color="#1f77b4", lw=2, label="smoothed")

    plt.axhline(enter_thr, color="#d62728", lw=1, ls="--", alpha=0.7, label="enter (bottom)")
    plt.axhline(leave_thr, color="#ff7f0e", lw=1, ls="--", alpha=0.7, label="leave (bottom)")
    plt.axhline(stand_thr, color="#2ca02c", lw=1, ls="--", alpha=0.7, label="standing")

    plt.plot(times[peaks], smooth[peaks], "rv", ms=12, label=f"find peaks reps ({len(peaks)})")
    plt.plot(times[sm_reps], smooth[sm_reps], "go", ms=10, label=f"state machine reps ({len(sm_reps)})")

    plt.gca().invert_yaxis()                  # "up in the room" points up
    plt.xlabel("time (s)")
    plt.ylabel("body height (normalized, inverted)")
    plt.title(f"find peaks: {len(peaks)}   |   state machine: {len(sm_reps)}")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    print(f"Saved verification plot -> {out_path}")


def render(video_path, out_path, landmarks, rep_frames, fps, src_size):
    """Write the annotated video: skeleton + running counter + rep feedback.

    Reuses the cached `landmarks` (no second pose pass). `rep_frames` are the
    frame indices where a rep was counted.
    """
    src_w, src_h = src_size
    if OUTPUT_HEIGHT and OUTPUT_HEIGHT != src_h:
        scale = OUTPUT_HEIGHT / src_h         # shrink for a small, fast file
        out_w, out_h = int(round(src_w * scale)), OUTPUT_HEIGHT
    else:
        out_w, out_h = src_w, src_h           # keep source resolution
    flash_frames = int(FLASH_SECONDS * fps)   # frames the badge stays visible

    cap = cv2.VideoCapture(video_path)
    # "mp4v" is a widely-supported codec; the wrapper transcodes to H.264 after.
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))

    # Scale the overlay from the output height, while keeping it usable on
    # unusually small or large videos. All dimensions below derive from this.
    ui = max(0.65, min(2.0, out_h / 720.0))
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
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                    label_scale, label_thick)

    def dim_panel(x0, y0, x1, y1, color=OVERLAY_BG, opacity=0.72):
        """Blend a clipped translucent panel into the frame."""
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(out_w, x1), min(out_h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        roi = frame[y0:y1, x0:x1]
        tint = np.full_like(roi, color)
        cv2.addWeighted(tint, opacity, roi, 1.0 - opacity, 0, roi)

    progress = make_progress("render", len(landmarks))
    count = 0
    last_rep_frame = None
    i = -1
    for i in range(len(landmarks)):
        ok, frame = cap.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
            frame = cv2.resize(frame, (out_w, out_h))   # landmarks are normalized

        # Skeleton (only if a pose was found on this frame).
        if landmarks[i] is not None:
            mp_draw.draw_landmarks(
                frame, landmarks[i], mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

        # Advance through the already-sorted completion frames once. Besides
        # being cheaper than scanning every rep on every frame, this gives us
        # the exact age of the most recent rep for the feedback fade.
        while count < len(rep_frames) and rep_frames[count] <= i:
            last_rep_frame = rep_frames[count]
            count += 1

        # Compact counter card: neutral text for sustained readability and a
        # green accent reserved for success/confirmation.
        count_text = str(count)
        count_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX,
                                        count_scale, count_thick)
        content_h = max(label_size[1], count_size[1])
        card_x0, card_y0 = margin, margin
        card_w = accent_w + pad_x * 3 + label_size[0] + count_size[0]
        card_h = content_h + pad_y * 2
        card_x1, card_y1 = card_x0 + card_w, card_y0 + card_h
        dim_panel(card_x0, card_y0, card_x1, card_y1)
        cv2.rectangle(frame, (card_x0, card_y0),
                      (card_x0 + accent_w, card_y1), OVERLAY_ACCENT, -1)

        baseline = card_y0 + pad_y + content_h
        label_x = card_x0 + accent_w + pad_x
        cv2.putText(frame, label, (label_x, baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, OVERLAY_MUTED,
                    label_thick, cv2.LINE_AA)
        count_x = label_x + label_size[0] + pad_x
        cv2.putText(frame, count_text, (count_x, baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, count_scale, OVERLAY_TEXT,
                    count_thick, cv2.LINE_AA)

        # A confirmation chip appears beside the counter and fades smoothly.
        # If the frame is too narrow, it moves below the card automatically.
        if last_rep_frame is not None and flash_frames > 0:
            rep_age = i - last_rep_frame
            if 0 <= rep_age < flash_frames:
                progress_through_flash = rep_age / max(1, flash_frames - 1)
                fade = 1.0 if progress_through_flash < 0.45 else max(
                    0.0, (1.0 - progress_through_flash) / 0.55)
                chip_text = "+1 REP"
                chip_size, _ = cv2.getTextSize(
                    chip_text, cv2.FONT_HERSHEY_SIMPLEX, chip_scale, chip_thick)
                chip_w = chip_size[0] + pad_x * 2
                chip_h = max(chip_size[1] + pad_y * 2, card_h)
                chip_x0, chip_y0 = card_x1 + gap, card_y0
                if chip_x0 + chip_w > out_w - margin:
                    chip_x0, chip_y0 = card_x0, card_y1 + gap
                chip_x1, chip_y1 = chip_x0 + chip_w, chip_y0 + chip_h

                overlay = frame.copy()
                cv2.rectangle(overlay, (chip_x0, chip_y0),
                              (chip_x1, chip_y1), OVERLAY_ACCENT, -1)
                chip_baseline = chip_y0 + (chip_h + chip_size[1]) // 2
                cv2.putText(overlay, chip_text,
                            (chip_x0 + pad_x, chip_baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, chip_scale, (15, 15, 15),
                            chip_thick, cv2.LINE_AA)
                cv2.addWeighted(overlay, fade, frame, 1.0 - fade, 0, frame)

        writer.write(frame)
        progress(i)
    progress(i, force=True)   # ensure a clean final newline even on early break
    cap.release()
    writer.release()
    print(f"Saved annotated video -> {out_path}")


def main():
    # Only the parameters that change the RESULT are flags: the detection
    # knobs (for both methods) and the two output paths. Cosmetic settings
    # live as constants above.
    ap = argparse.ArgumentParser(
        description="Count burpees two ways (find_peaks vs state machine) + render annotated video")
    ap.add_argument("video", help="path to input video")
    ap.add_argument("--out", default="annotated.mp4", help="annotated video output")
    ap.add_argument("--plot", default="signal.png", help="verification plot output")
    ap.add_argument("--prominence", type=float, default=0.04,
                    help="[find_peaks] min peak prominence (normalized units)")
    ap.add_argument("--depth-frac", type=float, default=0.5,
                    help="[find_peaks] how far a dip must reach toward the ground [0-1]")
    ap.add_argument("--min-gap", type=float, default=1.0,
                    help="min seconds between reps (both methods)")
    ap.add_argument("--enter-frac", type=float, default=0.6,
                    help="[state machine] fraction of range to enter BOTTOM")
    ap.add_argument("--leave-frac", type=float, default=0.4,
                    help="[state machine] fraction of range to leave BOTTOM (< enter-frac, hysteresis)")
    ap.add_argument("--stand-frac", type=float, default=0.25,
                    help="[state machine] fraction of range to be considered STANDING again")
    ap.add_argument("--frame-step", type=int, default=1,
                    help="process every Nth frame for pose (1=every frame; higher=faster)")
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2],
                    help="MediaPipe pose model: 0=lite/fast, 1=full (default), 2=heavy")
    ap.add_argument("--no-video", action="store_true",
                    help="skip the annotated video render (count + plot only, fastest)")
    args = ap.parse_args()

    # 1-2. Pose + body-height signal (single pass).
    fps, size, landmarks, valid_idx, values = analyze(
        args.video, args.frame_step, args.model_complexity)
    if not values:
        sys.exit("No pose detected in any frame — check camera framing.")

    # 3. Count reps — both methods share the same smoothed signal so the
    # comparison isn't skewed by different preprocessing.
    times = np.array(valid_idx) / fps
    values = np.array(values)
    smooth, dt = smooth_signal(times, values)

    count, peaks = count_reps(times, values, smooth, dt,
                              args.prominence, args.min_gap, args.depth_frac)
    sm_count, sm_reps, sm_phase = count_reps_state_machine(
        times, smooth, args.enter_frac, args.leave_frac,
        args.stand_frac, args.min_gap)

    rep_frames = sorted(valid_idx[p] for p in peaks)         # find_peaks -> frames
    sm_rep_frames = sorted(valid_idx[p] for p in sm_reps)    # state machine -> frames

    # Report to the terminal.
    print("\n" + "=" * 50)
    print(f"  find_peaks    : {count} reps")
    print(f"  state machine : {sm_count} reps")
    print("=" * 50)
    if rep_frames:
        print("find_peaks timestamps:    " +
              ", ".join(f"{rf / fps:.1f}s" for rf in rep_frames))
    if sm_rep_frames:
        print("state machine timestamps: " +
              ", ".join(f"{rf / fps:.1f}s" for rf in sm_rep_frames))

    # 4. Verification plot (both methods overlaid) + annotated video, driven
    # by the state machine's reps.
    lo, hi = np.percentile(smooth, 5), np.percentile(smooth, 95)
    span = hi - lo
    thresholds = (lo + args.enter_frac * span, lo + args.leave_frac * span,
                 lo + args.stand_frac * span)
    save_plot(times, values, smooth, peaks, sm_reps, sm_phase, thresholds, args.plot)
    if not args.no_video:
        render(args.video, args.out, landmarks, sm_rep_frames, fps, size)


if __name__ == "__main__":
    main()
