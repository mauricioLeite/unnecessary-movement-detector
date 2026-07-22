# Unnecessary Movement Detector
#   make build                    build the image
#   make count VIDEO=clip.mp4     count repetitions and plot the signal
#   make run   VIDEO=clip.mp4     also render the annotated video
#   make test                     run the unit tests
#   make clean OUTDIR=out         remove an output directory
#   options: OUTDIR=dir FLAGS="--frame-step 2 --model-complexity 0"

# Lowercase overrides are accepted for compatibility.
VIDEO   ?= $(or $(video),example.mp4)
OUTDIR  ?= $(or $(outdir),out)
FLAGS   ?= $(flags)

VIDEO_PATH := $(abspath $(VIDEO))
VIDEO_NAME := $(notdir $(VIDEO))
OUT_DIR    := $(abspath $(OUTDIR))
STEM       := $(basename $(VIDEO_NAME))
# Not UID/GID: shells reset those, so they never reach compose.
HOST_UID   := $(shell id -u)
HOST_GID   := $(shell id -g)

export VIDEO_PATH VIDEO_NAME OUT_DIR STEM FLAGS HOST_UID HOST_GID

COMPOSE ?= docker compose
# -T leaves stdout non-interactive, so the script reports progress line by line.
RUN      = $(COMPOSE) run --rm -T

.PHONY: build run count test clean help _check

help:
	@sed -n 's/^# \{0,1\}//; 1,/^$$/p' $(MAKEFILE_LIST)

build:
	$(COMPOSE) build

# Docker treats a missing bind-mount source as a directory.
_check:
	@test -f "$(VIDEO_PATH)" || { \
		echo "ERROR: video not found: '$(VIDEO)' (looked at $(VIDEO_PATH))"; \
		echo "       pass one with:  make run VIDEO=your_clip.mp4"; \
		exit 1; }

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

run: _check | $(OUT_DIR)
	$(RUN) render
	$(RUN) transcode
	-$(RUN) fix-perms
	rm -f $(OUT_DIR)/$(STEM)_raw.mp4
	@echo ">> Done. See $(OUTDIR)/$(STEM)_annotated.mp4 and $(OUTDIR)/$(STEM)_signal.png"

count: _check | $(OUT_DIR)
	$(RUN) count
	-$(RUN) fix-perms
	@echo ">> Done (count only). See $(OUTDIR)/$(STEM)_signal.png"

test:
	$(RUN) test

clean:
	rm -rf $(OUT_DIR)
