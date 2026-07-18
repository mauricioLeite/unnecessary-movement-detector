"""Count burpees in a video and render an annotated version — one pass.

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
  3. COUNT    Smooth the signal and count the ground-contact dips with
              scipy.find_peaks.
  4. OUTPUTS  Print the count, save a verification plot, and render an
              annotated video (skeleton + live counter + REP! flash).

We run pose estimation ONCE and reuse the cached landmarks for both the count
and the annotated video.
=============================================================================
"""

import argparse
import sys

import cv2                                    # read/write video, draw overlays
import numpy as np                            # numeric arrays / math
import mediapipe as mp                        # pretrained pose estimator
from scipy.signal import find_peaks, savgol_filter

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils          # draws the skeleton
mp_styles = mp.solutions.drawing_styles

# --- Fixed settings (edit here; intentionally NOT command-line flags) -------
OUTPUT_HEIGHT = 960       # output video height in px; width scales to keep ratio
SMOOTH_SECONDS = 0.4      # smoothing window as a real duration (fps-independent)
FLASH_SECONDS = 0.5       # how long the "REP!" badge stays on after each rep
REP_Y = 0.15              # vertical position of the REP! badge (0=top, 1=bottom)


def analyze(video_path):
    """Single pose pass over the video.

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
    with mp_pose.Pose(model_complexity=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # OpenCV gives BGR; MediaPipe wants RGB.
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                landmarks.append(res.pose_landmarks)
                # Body-height = mean y of confidently-visible joints. Ignoring
                # low-visibility joints keeps a half-guessed limb from skewing it.
                ys = [lm.y for lm in res.pose_landmarks.landmark
                      if lm.visibility > 0.5]
                if ys:
                    valid_idx.append(i)
                    values.append(float(np.mean(ys)))
            else:
                landmarks.append(None)   # no person detected this frame
            if i % 100 == 0:
                print(f"  pose {i}/{total}", end="\r")
            i += 1
    cap.release()
    print()
    return fps, (w, h), landmarks, valid_idx, values


def count_reps(times, values, prominence, min_gap_s, depth_frac):
    """Count the ground-contact dips in the body-height signal.

    Returns (count, peak_indices, smoothed_signal). peak_indices index into
    times/values at each detected rep.
    """
    if len(values) < 5:
        return 0, np.array([]), values

    # Typical spacing (seconds) between samples — used for the smoothing window
    # and to convert the min rep gap into a number of samples.
    dt = np.median(np.diff(times)) if len(times) > 1 else 1 / 30

    # --- Smooth ---
    # Savitzky-Golay fits a small polynomial over a sliding window, removing
    # jitter while preserving dip shape/depth. Window is sized from a real
    # duration so it's the same regardless of fps. It must be odd and >= 5.
    win = int(round(SMOOTH_SECONDS / dt))
    if win % 2 == 0:
        win += 1
    win = max(win, 5)
    if win > len(values):
        win = (len(values) - 1) | 1
    smooth = savgol_filter(values, win, 3) if win >= 5 else np.asarray(values)

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
    return len(peaks), peaks, smooth


def save_plot(times, raw, smooth, peaks, out_path):
    """Save the body-height wave with detected reps marked, for eyeball checks."""
    import matplotlib
    matplotlib.use("Agg")                     # headless backend (no display)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 5))
    plt.plot(times, raw, color="#bbb", lw=1, label="raw centroid height")
    plt.plot(times, smooth, color="#1f77b4", lw=2, label="smoothed")
    plt.plot(times[peaks], smooth[peaks], "rv", ms=12,
             label=f"counted reps ({len(peaks)})")
    plt.gca().invert_yaxis()                  # "up in the room" points up
    plt.xlabel("time (s)")
    plt.ylabel("body height (normalized, inverted)")
    plt.title(f"Burpee count: {len(peaks)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    print(f"Saved verification plot -> {out_path}")


def render(video_path, out_path, landmarks, rep_frames, fps, src_size):
    """Write the annotated video: skeleton + running counter + REP! flash.

    Reuses the cached `landmarks` (no second pose pass). `rep_frames` are the
    frame indices where a rep was counted.
    """
    src_w, src_h = src_size
    scale = OUTPUT_HEIGHT / src_h             # shrink for a small, fast file
    out_w, out_h = int(round(src_w * scale)), OUTPUT_HEIGHT
    flash_frames = int(FLASH_SECONDS * fps)   # frames the badge stays visible

    cap = cv2.VideoCapture(video_path)
    # "mp4v" is a widely-supported codec; the wrapper transcodes to H.264 after.
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))

    for i in range(len(landmarks)):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (out_w, out_h))   # landmarks are normalized

        # Skeleton (only if a pose was found on this frame).
        if landmarks[i] is not None:
            mp_draw.draw_landmarks(
                frame, landmarks[i], mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

        count = sum(1 for rf in rep_frames if rf <= i)          # running total
        recent_rep = any(0 <= i - rf < flash_frames for rf in rep_frames)

        # Counter banner (top strip). cv2 colors are BGR: (0,255,0)=green.
        cv2.rectangle(frame, (0, 0), (out_w, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"BURPEES: {count}", (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)

        # REP! flash — a self-contained badge (translucent dark box + centered
        # text) so it stays legible over any background and any aspect ratio.
        # It sits high in the frame, where the athlete never is mid-rep.
        if recent_rep:
            scale_txt = out_h / 430.0
            thick = max(2, int(round(scale_txt * 2.5)))
            (tw, th), _ = cv2.getTextSize("REP!", cv2.FONT_HERSHEY_SIMPLEX,
                                          scale_txt, thick)
            cx, cy = out_w // 2, int(out_h * REP_Y)
            pad = int(th * 0.6)
            x0, y0 = max(0, cx - tw // 2 - pad), max(0, cy - th // 2 - pad)
            x1, y1 = min(out_w, cx + tw // 2 + pad), min(out_h, cy + th // 2 + pad)
            roi = frame[y0:y1, x0:x1]                    # dim the background box
            cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, roi)
            cv2.putText(frame, "REP!", (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, scale_txt, (0, 200, 255),
                        thick, cv2.LINE_AA)

        writer.write(frame)
        if i % 100 == 0:
            print(f"  render {i}/{len(landmarks)}", end="\r")
    cap.release()
    writer.release()
    print(f"\nSaved annotated video -> {out_path}")


def main():
    # Only the parameters that change the RESULT are flags: the three detection
    # knobs and the two output paths. Cosmetic settings live as constants above.
    ap = argparse.ArgumentParser(description="Count burpees + render annotated video")
    ap.add_argument("video", help="path to input video")
    ap.add_argument("--out", default="annotated.mp4", help="annotated video output")
    ap.add_argument("--plot", default="signal.png", help="verification plot output")
    ap.add_argument("--prominence", type=float, default=0.04,
                    help="min peak prominence (normalized units)")
    ap.add_argument("--min-gap", type=float, default=1.0,
                    help="min seconds between reps")
    ap.add_argument("--depth-frac", type=float, default=0.5,
                    help="how far a dip must reach toward the ground [0-1]")
    args = ap.parse_args()

    # 1-2. Pose + body-height signal (single pass).
    fps, size, landmarks, valid_idx, values = analyze(args.video)
    if not values:
        sys.exit("No pose detected in any frame — check camera framing.")

    # 3. Count reps.
    times = np.array(valid_idx) / fps
    count, peaks, smooth = count_reps(times, np.array(values),
                                      args.prominence, args.min_gap, args.depth_frac)
    rep_frames = sorted(valid_idx[p] for p in peaks)   # peaks -> frame indices

    # Report to the terminal.
    print("\n" + "=" * 40)
    print(f"  BURPEE COUNT: {count}")
    print("=" * 40)
    if rep_frames:
        print("Rep timestamps: " +
              ", ".join(f"{rf / fps:.1f}s" for rf in rep_frames))

    # 4. Verification plot + annotated video.
    save_plot(times, np.array(values), smooth, peaks, args.plot)
    render(args.video, args.out, landmarks, rep_frames, fps, size)


if __name__ == "__main__":
    main()
