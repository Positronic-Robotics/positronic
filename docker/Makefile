.PHONY: all build tag push clean prune help build-training tag-training push-training build-openpi tag-openpi push-openpi build-groot tag-groot push-groot

# Image configuration
IMAGE_NAME_TRAINING := positro/positronic
IMAGE_NAME_OPENPI := positro/openpi
IMAGE_NAME_GROOT := positro/gr00t
IMAGE_NAME_OPENPI_BASE := positro/openpi-base

# Extract version from pyproject.toml (first literal version entry)
VERSION := $(shell sed -n 's/^version = "\([^"]*\)"/\1/p' ../pyproject.toml | head -n 1)

# Get git commit SHA (short)
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Branch name sanitized for Docker tags (replace / with -)
BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null | tr '/' '-')

# IMAGE_TAG: branch name locally, overridable via `make push IMAGE_TAG=foo`
IMAGE_TAG ?= $(BRANCH)

REGISTRY_URL ?= docker.io

# Local build tags
LOCAL_TAG_TRAINING := $(IMAGE_NAME_TRAINING):local

# OpenPI base image (pulled from registry)
TAG_OPENPI_BASE := $(IMAGE_NAME_OPENPI_BASE):latest

help:
	@echo "Positronic Docker Build System - Makefile"
	@echo ""
	@echo "Configuration:"
	@echo "  Training Image: $(IMAGE_NAME_TRAINING)"
	@echo "  Version:        $(VERSION)"
	@echo "  Git SHA:        $(GIT_SHA)"
	@echo "  Image Tag:      $(IMAGE_TAG)"
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
	@echo ""
	@echo "Override image tag: make push IMAGE_TAG=my-feature"

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

build: build-training build-openpi build-groot

tag-training: build-training
	@echo "Tagging training image..."
	docker tag $(LOCAL_TAG_TRAINING) $(IMAGE_NAME_TRAINING):$(IMAGE_TAG)
	docker tag $(LOCAL_TAG_TRAINING) $(IMAGE_NAME_TRAINING):$(GIT_SHA)
ifdef CI
	docker tag $(LOCAL_TAG_TRAINING) $(IMAGE_NAME_TRAINING):latest
	docker tag $(LOCAL_TAG_TRAINING) $(IMAGE_NAME_TRAINING):v$(VERSION)
endif
	@echo "Tagged: $(IMAGE_TAG), $(GIT_SHA)"

tag-openpi: build-openpi
	@echo "Tagging openpi image..."
	docker tag $(IMAGE_NAME_OPENPI):local $(IMAGE_NAME_OPENPI):$(IMAGE_TAG)
	docker tag $(IMAGE_NAME_OPENPI):local $(IMAGE_NAME_OPENPI):$(GIT_SHA)
ifdef CI
	docker tag $(IMAGE_NAME_OPENPI):local $(IMAGE_NAME_OPENPI):latest
endif
	@echo "Tagged: $(IMAGE_TAG), $(GIT_SHA)"

tag-groot: build-groot
	@echo "Tagging groot image..."
	docker tag $(IMAGE_NAME_GROOT):local $(IMAGE_NAME_GROOT):$(IMAGE_TAG)
	docker tag $(IMAGE_NAME_GROOT):local $(IMAGE_NAME_GROOT):$(GIT_SHA)
ifdef CI
	docker tag $(IMAGE_NAME_GROOT):local $(IMAGE_NAME_GROOT):latest
endif
	@echo "Tagged: $(IMAGE_TAG), $(GIT_SHA)"

tag: tag-training tag-openpi tag-groot

push-training: tag-training
	@echo "Pushing training images to Docker Hub..."
	docker push $(IMAGE_NAME_TRAINING):$(IMAGE_TAG)
	docker push $(IMAGE_NAME_TRAINING):$(GIT_SHA)
ifdef CI
	docker push $(IMAGE_NAME_TRAINING):latest
	docker push $(IMAGE_NAME_TRAINING):v$(VERSION)
endif
	@echo "IMAGE_TAG=$(IMAGE_TAG)" > .env
	@echo ""
	@echo "Successfully pushed training images!"

push-openpi: tag-openpi
	@echo "Pushing $(IMAGE_NAME_OPENPI) to Docker Hub..."
	docker push $(IMAGE_NAME_OPENPI):$(IMAGE_TAG)
	docker push $(IMAGE_NAME_OPENPI):$(GIT_SHA)
ifdef CI
	docker push $(IMAGE_NAME_OPENPI):latest
endif
	@echo "IMAGE_TAG=$(IMAGE_TAG)" > .env

push-groot: tag-groot
	@echo "Pushing $(IMAGE_NAME_GROOT) to Docker Hub..."
	docker push $(IMAGE_NAME_GROOT):$(IMAGE_TAG)
	docker push $(IMAGE_NAME_GROOT):$(GIT_SHA)
ifdef CI
	docker push $(IMAGE_NAME_GROOT):latest
endif
	@echo "IMAGE_TAG=$(IMAGE_TAG)" > .env

push: push-training push-openpi push-groot

all: push

clean:
	@echo "Removing all local images..."
	-docker rmi $(LOCAL_TAG_TRAINING)
	-docker rmi $(IMAGE_NAME_TRAINING):$(IMAGE_TAG)
	-docker rmi $(IMAGE_NAME_TRAINING):$(GIT_SHA)
	-docker rmi $(IMAGE_NAME_OPENPI):local
	-docker rmi $(IMAGE_NAME_OPENPI):$(IMAGE_TAG)
	-docker rmi $(IMAGE_NAME_OPENPI):$(GIT_SHA)
	-docker rmi $(IMAGE_NAME_GROOT):local
	-docker rmi $(IMAGE_NAME_GROOT):$(IMAGE_TAG)
	-docker rmi $(IMAGE_NAME_GROOT):$(GIT_SHA)
	@echo "All local images removed."

prune:
	@echo "Pruning dangling and unused Docker images..."
	docker image prune -f
	@echo "Prune complete."
