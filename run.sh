#!/usr/bin/env bash
#
# Burpee counter wrapper — count + annotated video in one command.
#
# Usage:
#   ./run.sh VIDEO.mp4 [OUTDIR] [-- extra flags passed to the scripts]
#
# Examples:
#   ./run.sh example.mp4
#   ./run.sh workout.mp4 results
#   ./run.sh hard_clip.mp4 out -- --depth-frac 0.4 --min-gap 0.8
#
set -euo pipefail

IMAGE="burpee-counter"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 VIDEO [OUTDIR] [-- extra flags]" >&2
  exit 1
fi

VIDEO="$1"; shift
OUTDIR="."
if [[ $# -gt 0 && "$1" != "--" ]]; then OUTDIR="$1"; shift; fi
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")   # tuning flags shared by both scripts

if [[ ! -f "$VIDEO" ]]; then echo "No such file: $VIDEO" >&2; exit 1; fi

VIDEO_ABS="$(realpath "$VIDEO")"
VIDEO_NAME="$(basename "$VIDEO")"
mkdir -p "$OUTDIR"
OUTDIR_ABS="$(realpath "$OUTDIR")"
STEM="${VIDEO_NAME%.*}"

# Build the image if it isn't present yet.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">> Building Docker image ($IMAGE)..."
  docker build -t "$IMAGE" .
fi

run() {  # run <entrypoint> <args...>
  docker run --rm --entrypoint "$1" \
    -v "$VIDEO_ABS:/data/$VIDEO_NAME:ro" \
    -v "$OUTDIR_ABS:/out" \
    "$IMAGE" "${@:2}"
}

echo ">> [1/2] Counting + rendering (single pose pass)..."
run python burpee_counter.py "/data/$VIDEO_NAME" \
    --out "/out/${STEM}_raw.mp4" --plot "/out/${STEM}_signal.png" "${EXTRA[@]}"

echo ">> [2/2] Transcoding to H.264..."
run ffmpeg -y -i "/out/${STEM}_raw.mp4" \
    -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
    "/out/${STEM}_annotated.mp4" >/dev/null 2>&1

# Clean up intermediate + fix ownership of outputs (Docker writes as root).
run rm -f "/out/${STEM}_raw.mp4"
run chown -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true

echo ""
echo ">> Done. Outputs in '$OUTDIR':"
echo "     ${STEM}_annotated.mp4   (skeleton + live counter)"
echo "     ${STEM}_signal.png      (verification plot)"
