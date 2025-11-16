.PHONY: all build tag push clean prune help build-base build-training build-groot tag-training tag-groot push-training push-groot

# Image configuration
IMAGE_NAME_TRAINING := positronic-training
IMAGE_NAME_GROOT := positronic-groot
IMAGE_NAME_BASE := positronic-base

# Extract version from pyproject.toml
VERSION := $(shell grep '^version = ' ../pyproject.toml | sed 's/version = "\(.*\)"/\1/')

# Get git commit SHA (short)
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Get Nebius configuration from CLI
# Extract registry info from JSON format
REGISTRY_INFO := $(shell nebius registry list --format json 2>/dev/null | jq -r '.items[] | select(.metadata.name=="positronic") | .metadata.id + " " + .status.registry_fqdn' || echo "")
NB_REGISTRY_PATH := $(shell echo "$(REGISTRY_INFO)" | cut -d' ' -f1 | cut -d- -f2)
NB_REGISTRY_FQDN := $(shell echo "$(REGISTRY_INFO)" | cut -d' ' -f2)

# Full registry URL
REGISTRY_URL := $(NB_REGISTRY_FQDN)/$(NB_REGISTRY_PATH)

# Image tags for training
TAG_TRAINING_LATEST := $(REGISTRY_URL)/$(IMAGE_NAME_TRAINING):latest
TAG_TRAINING_VERSION := $(REGISTRY_URL)/$(IMAGE_NAME_TRAINING):v$(VERSION)
TAG_TRAINING_SHA := $(REGISTRY_URL)/$(IMAGE_NAME_TRAINING):$(GIT_SHA)
LOCAL_TAG_TRAINING := $(IMAGE_NAME_TRAINING):local

# Image tags for groot
TAG_GROOT_LATEST := $(REGISTRY_URL)/$(IMAGE_NAME_GROOT):latest
TAG_GROOT_VERSION := $(REGISTRY_URL)/$(IMAGE_NAME_GROOT):v$(VERSION)
TAG_GROOT_SHA := $(REGISTRY_URL)/$(IMAGE_NAME_GROOT):$(GIT_SHA)
LOCAL_TAG_GROOT := $(IMAGE_NAME_GROOT):local

# Local base image tag
LOCAL_TAG_BASE := $(IMAGE_NAME_BASE):local

help:
	@echo "Positronic Docker Build System - Makefile"
	@echo ""
	@echo "Configuration:"
	@echo "  Training Image: $(IMAGE_NAME_TRAINING)"
	@echo "  Groot Image:    $(IMAGE_NAME_GROOT)"
	@echo "  Base Image:     $(IMAGE_NAME_BASE)"
	@echo "  Version:        $(VERSION)"
	@echo "  Git SHA:        $(GIT_SHA)"
	@echo "  Registry FQDN:  $(NB_REGISTRY_FQDN)"
	@echo "  Registry Path:  $(NB_REGISTRY_PATH)"
	@echo "  Registry URL:   $(REGISTRY_URL)"
	@echo ""
	@echo "Targets:"
	@echo "  make build-base      Build the base image with common dependencies"
	@echo "  make build-training  Build the training image (depends on base)"
	@echo "  make build-groot     Build the groot image (depends on base)"
	@echo "  make build           Build base, training, and groot images"
	@echo ""
	@echo "  make tag-training    Tag the training image (depends on build-training)"
	@echo "  make tag-groot       Tag the groot image (depends on build-groot)"
	@echo "  make tag             Tag both images"
	@echo ""
	@echo "  make push-training   Push training tags to Nebius (depends on tag-training)"
	@echo "  make push-groot      Push groot tags to Nebius (depends on tag-groot)"
	@echo "  make push            Push all tags to Nebius (depends on tag)"
	@echo ""
	@echo "  make all             Build, tag, and push all images (alias for push)"
	@echo "  make clean           Remove all local images"
	@echo "  make prune           Remove dangling/unused Docker images"
	@echo "  make help            Show this help message"

build-base:
	@echo "Building base image with common dependencies..."
	docker build --platform linux/amd64 -f Dockerfile.base -t $(LOCAL_TAG_BASE) ..

build-training: build-base
	@echo "Building $(IMAGE_NAME_TRAINING) with source code baked in..."
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: Could not extract version from pyproject.toml"; \
		exit 1; \
	fi
	docker build --platform linux/amd64 -f Dockerfile.training.cloud -t $(LOCAL_TAG_TRAINING) ..

build-groot: build-base
	@echo "Building $(IMAGE_NAME_GROOT) image..."
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: Could not extract version from pyproject.toml"; \
		exit 1; \
	fi
	docker build --platform linux/amd64 -f Dockerfile.groot -t $(LOCAL_TAG_GROOT) ../../gr00t

build: build-base build-training build-groot

tag-training: build-training
	@echo "Tagging training image with multiple tags..."
	@if [ -z "$(NB_REGISTRY_FQDN)" ] || [ -z "$(NB_REGISTRY_PATH)" ]; then \
		echo "Error: Nebius configuration not found. Please ensure:"; \
		echo "  1. Nebius CLI is configured (run: nebius config list)"; \
		echo "  2. Registry 'positronic' exists (run: nebius registry list)"; \
		exit 1; \
	fi
	docker tag $(LOCAL_TAG_TRAINING) $(TAG_TRAINING_LATEST)
	docker tag $(LOCAL_TAG_TRAINING) $(TAG_TRAINING_VERSION)
	docker tag $(LOCAL_TAG_TRAINING) $(TAG_TRAINING_SHA)
	@echo "Tagged with:"
	@echo "  - $(TAG_TRAINING_LATEST)"
	@echo "  - $(TAG_TRAINING_VERSION)"
	@echo "  - $(TAG_TRAINING_SHA)"

tag-groot: build-groot
	@echo "Tagging groot image with multiple tags..."
	@if [ -z "$(NB_REGISTRY_FQDN)" ] || [ -z "$(NB_REGISTRY_PATH)" ]; then \
		echo "Error: Nebius configuration not found. Please ensure:"; \
		echo "  1. Nebius CLI is configured (run: nebius config list)"; \
		echo "  2. Registry 'positronic' exists (run: nebius registry list)"; \
		exit 1; \
	fi
	docker tag $(LOCAL_TAG_GROOT) $(TAG_GROOT_LATEST)
	docker tag $(LOCAL_TAG_GROOT) $(TAG_GROOT_VERSION)
	docker tag $(LOCAL_TAG_GROOT) $(TAG_GROOT_SHA)
	@echo "Tagged with:"
	@echo "  - $(TAG_GROOT_LATEST)"
	@echo "  - $(TAG_GROOT_VERSION)"
	@echo "  - $(TAG_GROOT_SHA)"

tag: tag-training tag-groot

push-training: tag-training
	@echo "Pushing training images to Nebius Container Registry..."
	docker push $(TAG_TRAINING_LATEST)
	docker push $(TAG_TRAINING_VERSION)
	docker push $(TAG_TRAINING_SHA)
	@echo ""
	@echo "Successfully pushed training images to Nebius Container Registry!"
	@echo "To use on cloud instances, set:"
	@echo "  export IMAGE_REGISTRY=$(REGISTRY_URL)/"

push-groot: tag-groot
	@echo "Pushing groot images to Nebius Container Registry..."
	docker push $(TAG_GROOT_LATEST)
	docker push $(TAG_GROOT_VERSION)
	docker push $(TAG_GROOT_SHA)
	@echo ""
	@echo "Successfully pushed groot images to Nebius Container Registry!"
	@echo "To use on cloud instances, set:"
	@echo "  export IMAGE_REGISTRY=$(REGISTRY_URL)/"

push: push-training push-groot

all: push

clean:
	@echo "Removing all local images..."
	-docker rmi $(LOCAL_TAG_BASE)
	-docker rmi $(LOCAL_TAG_TRAINING)
	-docker rmi $(TAG_TRAINING_LATEST)
	-docker rmi $(TAG_TRAINING_VERSION)
	-docker rmi $(TAG_TRAINING_SHA)
	-docker rmi $(LOCAL_TAG_GROOT)
	-docker rmi $(TAG_GROOT_LATEST)
	-docker rmi $(TAG_GROOT_VERSION)
	-docker rmi $(TAG_GROOT_SHA)
	@echo "All local images removed."

prune:
	@echo "Pruning dangling and unused Docker images..."
	docker image prune -f
	@echo "Prune complete."
