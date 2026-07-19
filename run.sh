#!/usr/bin/env bash
#
# State-machine burpee counter — runs the modified script (find_peaks vs
# state machine, compared side by side) using the existing `burpee-counter`
# Docker image, without rebuilding it. The modified script is mounted over
# the image's copy, and video.mp4 in this folder is used as the input.
#
# Usage:
#   ./run.sh [-- extra flags passed to the script]
#
# Examples:
#   ./run.sh
#   ./run.sh -- --no-video                       # fast count-only pass
#   ./run.sh -- --enter-frac 0.65 --leave-frac 0.35
#
set -euo pipefail

IMAGE="burpee-counter"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO="$SCRIPT_DIR/video.mp4"
OUTDIR="$SCRIPT_DIR/out"
STEM="video"

[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")

NO_VIDEO=0
for flag in "${EXTRA[@]}"; do
  [[ "$flag" == "--no-video" ]] && NO_VIDEO=1
done

if [[ ! -f "$VIDEO" ]]; then echo "No such file: $VIDEO" >&2; exit 1; fi
mkdir -p "$OUTDIR"

# Build the image if it isn't present yet (same image the parent project uses).
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">> Building Docker image ($IMAGE)..."
  docker build -t "$IMAGE" "$SCRIPT_DIR/.."
fi

run() {  # run <entrypoint> <args...>
  docker run --rm --entrypoint "$1" \
    -v "$SCRIPT_DIR/burpee_counter.py:/app/burpee_counter.py:ro" \
    -v "$VIDEO:/data/video.mp4:ro" \
    -v "$OUTDIR:/out" \
    "$IMAGE" "${@:2}"
}

if [[ "$NO_VIDEO" -eq 1 ]]; then
  echo ">> Counting (state machine vs find_peaks, no video)..."
  run python burpee_counter.py "/data/video.mp4" \
      --plot "/out/${STEM}_compare.png" "${EXTRA[@]}"

  run chown -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true

  echo ""
  echo ">> Done. Outputs in 'out':"
  echo "     ${STEM}_compare.png      (find_peaks vs state machine plot)"
  exit 0
fi

echo ">> [1/2] Counting + rendering (state machine drives the video)..."
run python burpee_counter.py "/data/video.mp4" \
    --out "/out/${STEM}_raw.mp4" --plot "/out/${STEM}_compare.png" "${EXTRA[@]}"

echo ">> [2/2] Transcoding to H.264..."
run ffmpeg -y -i "/out/${STEM}_raw.mp4" \
    -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart \
    "/out/${STEM}_annotated.mp4" >/dev/null 2>&1

# Clean up intermediate + fix ownership of outputs (Docker writes as root).
run rm -f "/out/${STEM}_raw.mp4"
run chown -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true

echo ""
echo ">> Done. Outputs in 'out':"
echo "     ${STEM}_annotated.mp4   (skeleton + live counter, driven by state machine)"
echo "     ${STEM}_compare.png     (find_peaks vs state machine plot)"
