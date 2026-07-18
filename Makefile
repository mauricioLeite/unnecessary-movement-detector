# Burpee counter — convenience targets.
#
# Usage:
#   make build                       # build the Docker image (one time)
#   make run VIDEO=example.mp4       # count + plot + annotated video -> OUTDIR
#   make clean                       # remove generated outputs
#
# Override variables on the command line, e.g.:
#   make run VIDEO=workout.mp4 OUTDIR=results FLAGS="--min-gap 0.8"

IMAGE   ?= burpee-counter
# Accept either VIDEO=... or lowercase video=... (make vars are case-sensitive;
# supporting both avoids the common "my override was silently ignored" trap).
VIDEO   ?= $(or $(video),example.mp4)
OUTDIR  ?= $(or $(outdir),out)
FLAGS   ?= $(flags)

# Absolute paths + basename for clean docker -v mounts.
VIDEO_ABS  := $(abspath $(VIDEO))
VIDEO_NAME := $(notdir $(VIDEO))
OUTDIR_ABS := $(abspath $(OUTDIR))
STEM       := $(basename $(VIDEO_NAME))
UID        := $(shell id -u)
GID        := $(shell id -g)

# docker run with the input mounted read-only and OUTDIR writable.
DOCKER_RUN = docker run --rm \
	-v "$(VIDEO_ABS):/data/$(VIDEO_NAME):ro" \
	-v "$(OUTDIR_ABS):/out"

.PHONY: build run clean help _check

help:
	@sed -n 's/^# \{0,1\}//; 1,/^$$/p' $(MAKEFILE_LIST)

build:
	docker build -t $(IMAGE) .

# Fail early with a clear message if the input isn't a real file. Without this,
# Docker would silently create an empty DIRECTORY at the mount path and then
# OpenCV would report the confusing "Could not open video".
_check:
	@test -f "$(VIDEO_ABS)" || { \
		echo "ERROR: video not found: '$(VIDEO)' (looked at $(VIDEO_ABS))"; \
		echo "       pass one with:  make run VIDEO=your_clip.mp4"; \
		exit 1; }

$(OUTDIR_ABS):
	mkdir -p $(OUTDIR_ABS)

# One script run does count + plot + annotated (mp4v); then we transcode to
# H.264 for universal playback, drop the intermediate, and fix ownership.
run: _check | $(OUTDIR_ABS)
	$(DOCKER_RUN) $(IMAGE) /data/$(VIDEO_NAME) \
		--out /out/$(STEM)_raw.mp4 --plot /out/$(STEM)_signal.png $(FLAGS)
	$(DOCKER_RUN) --entrypoint ffmpeg $(IMAGE) -y -i /out/$(STEM)_raw.mp4 \
		-c:v libx264 -pix_fmt yuv420p -movflags +faststart \
		/out/$(STEM)_annotated.mp4
	$(DOCKER_RUN) --entrypoint rm $(IMAGE) -f /out/$(STEM)_raw.mp4
	-$(DOCKER_RUN) --entrypoint chown $(IMAGE) -R $(UID):$(GID) /out
	@echo ">> Done. See $(OUTDIR)/$(STEM)_annotated.mp4 and $(STEM)_signal.png"

clean:
	rm -rf $(OUTDIR_ABS)
