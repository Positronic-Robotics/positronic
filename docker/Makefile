.PHONY: all build tag push clean prune help build-training tag-training push-training build-openpi tag-openpi push-openpi build-groot tag-groot push-groot build-dreamzero tag-dreamzero push-dreamzero

# Image configuration
IMAGE_NAME_TRAINING := positro/positronic
IMAGE_NAME_OPENPI := positro/openpi
IMAGE_NAME_GROOT := positro/gr00t
IMAGE_NAME_DREAMZERO := positro/dreamzero
IMAGE_NAME_OPENPI_BASE := positro/openpi-base

# Extract version from pyproject.toml (first literal version entry)
VERSION := $(shell sed -n 's/^version = "\([^"]*\)"/\1/p' ../pyproject.toml | head -n 1)

# Get git commit SHA (short)
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

REGISTRY_URL ?= docker.io

# Image tags for training
TAG_TRAINING_LATEST := $(IMAGE_NAME_TRAINING):latest
TAG_TRAINING_VERSION := $(IMAGE_NAME_TRAINING):v$(VERSION)
TAG_TRAINING_SHA := $(IMAGE_NAME_TRAINING):$(GIT_SHA)
LOCAL_TAG_TRAINING := $(IMAGE_NAME_TRAINING):local

# Other image tags
TAG_OPENPI_LATEST := $(IMAGE_NAME_OPENPI):latest
TAG_GROOT_LATEST := $(IMAGE_NAME_GROOT):latest
TAG_DREAMZERO_LATEST := $(IMAGE_NAME_DREAMZERO):latest

# OpenPI base image (pulled from registry)
TAG_OPENPI_BASE := $(IMAGE_NAME_OPENPI_BASE):latest

help:
	@echo "Positronic Docker Build System - Makefile"
	@echo ""
	@echo "Configuration:"
	@echo "  Training Image: $(IMAGE_NAME_TRAINING)"
	@echo "  Version:        $(VERSION)"
	@echo "  Git SHA:        $(GIT_SHA)"
	@echo ""
	@echo "Targets:"
	@echo "  make build-training  Build the training image"
	@echo "  make build           Build training, openpi and groot images"
	@echo ""
	@echo "  make tag-training    Tag the training image (depends on build-training)"
	@echo "  make tag             Tag all images"
	@echo ""
	@echo "  make push-training   Push training tags to Docker Hub (depends on tag-training)"
	@echo "  make push            Push all tags to Docker Hub (depends on tag)"
	@echo ""
	@echo "  make all             Build, tag, and push all images (alias for push)"
	@echo "  make clean           Remove all local images"
	@echo "  make prune           Remove dangling/unused Docker images"
	@echo "  make help            Show this help message"

build-training:
	@echo "Building $(IMAGE_NAME_TRAINING) with source code baked in..."
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: Could not extract version from pyproject.toml"; \
		exit 1; \
	fi
	docker build --platform linux/amd64 -f Dockerfile.training.cloud -t $(LOCAL_TAG_TRAINING) ..

build-openpi:
	@echo "Building $(IMAGE_NAME_OPENPI) using base $(TAG_OPENPI_BASE)..."
	docker build --pull --platform linux/amd64 --build-arg BASE_IMAGE=$(TAG_OPENPI_BASE) -f Dockerfile.openpi -t $(IMAGE_NAME_OPENPI):local ..

build-groot: build-training
	@echo "Building $(IMAGE_NAME_GROOT)..."
	docker build --pull --platform linux/amd64 --build-arg BASE_IMAGE=positro/gr00t-base:latest -f Dockerfile.groot -t $(IMAGE_NAME_GROOT):local ..

build-dreamzero:
	@echo "Building $(IMAGE_NAME_DREAMZERO)..."
	docker build --platform linux/amd64 --build-arg BASE_IMAGE=positro/dreamzero-base:latest -f Dockerfile.dreamzero -t $(IMAGE_NAME_DREAMZERO):local ..

build: build-training build-openpi build-groot build-dreamzero

tag-training: build-training
	@echo "Tagging training image with multiple tags..."
	docker tag $(LOCAL_TAG_TRAINING) $(TAG_TRAINING_LATEST)
	docker tag $(LOCAL_TAG_TRAINING) $(TAG_TRAINING_VERSION)
	docker tag $(LOCAL_TAG_TRAINING) $(TAG_TRAINING_SHA)
	@echo "Tagged with:"
	@echo "  - $(TAG_TRAINING_LATEST)"
	@echo "  - $(TAG_TRAINING_VERSION)"
	@echo "  - $(TAG_TRAINING_SHA)"

tag: tag-training tag-openpi tag-groot tag-dreamzero

push-training: tag-training
	@echo "Pushing training images to Docker Hub..."
	docker push $(TAG_TRAINING_LATEST)
	docker push $(TAG_TRAINING_VERSION)
	docker push $(TAG_TRAINING_SHA)
	@echo ""
	@echo "Successfully pushed training images!"

push: push-training push-openpi push-groot push-dreamzero

tag-openpi: build-openpi
	@echo "Tagging openpi image..."
	docker tag $(IMAGE_NAME_OPENPI):local $(TAG_OPENPI_LATEST)
	@echo "Tagged with: $(TAG_OPENPI_LATEST)"

push-openpi: tag-openpi
	@echo "Pushing $(IMAGE_NAME_OPENPI) to Docker Hub..."
	docker push $(TAG_OPENPI_LATEST)

tag-groot: build-groot
	@echo "Tagging groot image..."
	docker tag $(IMAGE_NAME_GROOT):local $(TAG_GROOT_LATEST)
	@echo "Tagged with: $(TAG_GROOT_LATEST)"

push-groot: tag-groot
	@echo "Pushing $(IMAGE_NAME_GROOT) to Docker Hub..."
	docker push $(TAG_GROOT_LATEST)

tag-dreamzero: build-dreamzero
	@echo "Tagging dreamzero image..."
	docker tag $(IMAGE_NAME_DREAMZERO):local $(TAG_DREAMZERO_LATEST)
	@echo "Tagged with: $(TAG_DREAMZERO_LATEST)"

push-dreamzero: tag-dreamzero
	@echo "Pushing $(IMAGE_NAME_DREAMZERO) to Docker Hub..."
	docker push $(TAG_DREAMZERO_LATEST)

all: push

clean:
	@echo "Removing all local images..."
	-docker rmi $(LOCAL_TAG_TRAINING)
	-docker rmi $(TAG_TRAINING_LATEST)
	-docker rmi $(TAG_TRAINING_VERSION)
	-docker rmi $(TAG_TRAINING_SHA)
	-docker rmi $(IMAGE_NAME_OPENPI):local
	-docker rmi $(TAG_OPENPI_LATEST)
	-docker rmi $(IMAGE_NAME_GROOT):local
	-docker rmi $(TAG_GROOT_LATEST)
	-docker rmi $(IMAGE_NAME_DREAMZERO):local
	-docker rmi $(TAG_DREAMZERO_LATEST)
	@echo "All local images removed."

prune:
	@echo "Pruning dangling and unused Docker images..."
	docker image prune -f
	@echo "Prune complete."
