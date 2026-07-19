.PHONY: all build tag push clean prune help build-training tag-training push-training build-openpi tag-openpi push-openpi build-groot tag-groot push-groot build-dreamzero-base push-dreamzero-base build-dreamzero tag-dreamzero push-dreamzero build-robolab tag-robolab push-robolab nebius-login push-robolab-cr

# Image configuration
IMAGE_NAME_TRAINING := positro/positronic
IMAGE_NAME_OPENPI := positro/openpi
IMAGE_NAME_GROOT := positro/gr00t
IMAGE_NAME_DREAMZERO := positro/dreamzero
IMAGE_NAME_DREAMZERO_BASE := positro/dreamzero-base
IMAGE_NAME_OPENPI_BASE := positro/openpi-base
IMAGE_NAME_ROBOLAB := positro/robolab

# Extract version from pyproject.toml (first literal version entry)
VERSION := $(shell sed -n 's/^version = "\([^"]*\)"/\1/p' ../pyproject.toml | head -n 1)

# Get git commit SHA (short)
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Branch name sanitized for Docker tags (replace / with -)
BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null | tr '/' '-')

# IMAGE_TAG: branch name locally, overridable via `make push IMAGE_TAG=foo`
IMAGE_TAG ?= $(BRANCH)

REGISTRY_URL ?= docker.io

# Nebius Container Registry — opt-in path for an in-region pull from serverless
# jobs (the `push-*` targets above stay on Docker Hub and are unchanged). Override
# NEBIUS_REGISTRY for a different project/registry; set NEBIUS_PROFILE to use a
# non-default `nebius` CLI profile (e.g. a service-account profile for headless
# CI). Auth goes through the nebius credential helper — no registry secret here.
NEBIUS_REGISTRY ?= cr.eu-north1.nebius.cloud/e00a0ahqzcp9x0xczz
NEBIUS_REGISTRY_HOST := $(firstword $(subst /, ,$(NEBIUS_REGISTRY)))
NEBIUS_PROFILE_FLAG := $(if $(NEBIUS_PROFILE),--profile $(NEBIUS_PROFILE),)

# Local build tags
LOCAL_TAG_TRAINING := $(IMAGE_NAME_TRAINING):local

# Base image tags (not parameterized — these are heavy foundational images)
TAG_DREAMZERO_BASE_LATEST := $(IMAGE_NAME_DREAMZERO_BASE):latest

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

build-dreamzero-base:
	@echo "Building $(IMAGE_NAME_DREAMZERO_BASE)..."
	docker build --platform linux/amd64 -f ../positronic/vendors/dreamzero/Dockerfile -t $(TAG_DREAMZERO_BASE_LATEST) ..

push-dreamzero-base: build-dreamzero-base
	@echo "Pushing $(IMAGE_NAME_DREAMZERO_BASE) to Docker Hub..."
	docker push $(TAG_DREAMZERO_BASE_LATEST)

build-dreamzero:
	@echo "Building $(IMAGE_NAME_DREAMZERO)..."
	docker build --platform linux/amd64 --build-arg BASE_IMAGE=$(TAG_DREAMZERO_BASE_LATEST) -f Dockerfile.dreamzero -t $(IMAGE_NAME_DREAMZERO):local ..

build-robolab:
	@echo "Building $(IMAGE_NAME_ROBOLAB)..."
	docker build --platform linux/amd64 -f Dockerfile.robolab -t $(IMAGE_NAME_ROBOLAB):local ..

build: build-training build-openpi build-groot build-dreamzero build-robolab

tag-training: build-training
	@echo "Tagging training image..."
	docker tag $(LOCAL_TAG_TRAINING) $(IMAGE_NAME_TRAINING):$(IMAGE_TAG)
	docker tag $(LOCAL_TAG_TRAINING) $(IMAGE_NAME_TRAINING):$(GIT_SHA)
ifdef CI
	docker tag $(LOCAL_TAG_TRAINING) $(IMAGE_NAME_TRAINING):latest
endif
ifdef PUBLISH_RELEASE
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
ifdef PUBLISH_RELEASE
	docker tag $(IMAGE_NAME_OPENPI):local $(IMAGE_NAME_OPENPI):v$(VERSION)
endif
	@echo "Tagged: $(IMAGE_TAG), $(GIT_SHA)"

tag-groot: build-groot
	@echo "Tagging groot image..."
	docker tag $(IMAGE_NAME_GROOT):local $(IMAGE_NAME_GROOT):$(IMAGE_TAG)
	docker tag $(IMAGE_NAME_GROOT):local $(IMAGE_NAME_GROOT):$(GIT_SHA)
ifdef CI
	docker tag $(IMAGE_NAME_GROOT):local $(IMAGE_NAME_GROOT):latest
endif
ifdef PUBLISH_RELEASE
	docker tag $(IMAGE_NAME_GROOT):local $(IMAGE_NAME_GROOT):v$(VERSION)
endif
	@echo "Tagged: $(IMAGE_TAG), $(GIT_SHA)"

tag-dreamzero: build-dreamzero
	@echo "Tagging dreamzero image..."
	docker tag $(IMAGE_NAME_DREAMZERO):local $(IMAGE_NAME_DREAMZERO):$(IMAGE_TAG)
	docker tag $(IMAGE_NAME_DREAMZERO):local $(IMAGE_NAME_DREAMZERO):$(GIT_SHA)
ifdef CI
	docker tag $(IMAGE_NAME_DREAMZERO):local $(IMAGE_NAME_DREAMZERO):latest
endif
ifdef PUBLISH_RELEASE
	docker tag $(IMAGE_NAME_DREAMZERO):local $(IMAGE_NAME_DREAMZERO):v$(VERSION)
endif
	@echo "Tagged: $(IMAGE_TAG), $(GIT_SHA)"

tag-robolab: build-robolab
	@echo "Tagging robolab image..."
	docker tag $(IMAGE_NAME_ROBOLAB):local $(IMAGE_NAME_ROBOLAB):$(IMAGE_TAG)
	docker tag $(IMAGE_NAME_ROBOLAB):local $(IMAGE_NAME_ROBOLAB):$(GIT_SHA)
ifdef CI
	docker tag $(IMAGE_NAME_ROBOLAB):local $(IMAGE_NAME_ROBOLAB):latest
endif
ifdef PUBLISH_RELEASE
	docker tag $(IMAGE_NAME_ROBOLAB):local $(IMAGE_NAME_ROBOLAB):v$(VERSION)
endif
	@echo "Tagged: $(IMAGE_TAG), $(GIT_SHA)"

tag: tag-training tag-openpi tag-groot tag-dreamzero tag-robolab

push-training: tag-training
	@echo "Pushing training images to Docker Hub..."
	docker push $(IMAGE_NAME_TRAINING):$(IMAGE_TAG)
	docker push $(IMAGE_NAME_TRAINING):$(GIT_SHA)
ifdef CI
	docker push $(IMAGE_NAME_TRAINING):latest
endif
ifdef PUBLISH_RELEASE
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
ifdef PUBLISH_RELEASE
	docker push $(IMAGE_NAME_OPENPI):v$(VERSION)
endif
	@echo "IMAGE_TAG=$(IMAGE_TAG)" > .env

push-groot: tag-groot
	@echo "Pushing $(IMAGE_NAME_GROOT) to Docker Hub..."
	docker push $(IMAGE_NAME_GROOT):$(IMAGE_TAG)
	docker push $(IMAGE_NAME_GROOT):$(GIT_SHA)
ifdef CI
	docker push $(IMAGE_NAME_GROOT):latest
endif
ifdef PUBLISH_RELEASE
	docker push $(IMAGE_NAME_GROOT):v$(VERSION)
endif
	@echo "IMAGE_TAG=$(IMAGE_TAG)" > .env

push-dreamzero: tag-dreamzero
	@echo "Pushing $(IMAGE_NAME_DREAMZERO) to Docker Hub..."
	docker push $(IMAGE_NAME_DREAMZERO):$(IMAGE_TAG)
	docker push $(IMAGE_NAME_DREAMZERO):$(GIT_SHA)
ifdef CI
	docker push $(IMAGE_NAME_DREAMZERO):latest
endif
ifdef PUBLISH_RELEASE
	docker push $(IMAGE_NAME_DREAMZERO):v$(VERSION)
endif
	@echo "IMAGE_TAG=$(IMAGE_TAG)" > .env

push-robolab: tag-robolab
	@echo "Pushing $(IMAGE_NAME_ROBOLAB) to Docker Hub..."
	docker push $(IMAGE_NAME_ROBOLAB):$(IMAGE_TAG)
	docker push $(IMAGE_NAME_ROBOLAB):$(GIT_SHA)
ifdef CI
	docker push $(IMAGE_NAME_ROBOLAB):latest
endif
ifdef PUBLISH_RELEASE
	docker push $(IMAGE_NAME_ROBOLAB):v$(VERSION)
endif
	@echo "IMAGE_TAG=$(IMAGE_TAG)" > .env

push: push-training push-openpi push-groot push-dreamzero push-robolab

# Log docker/podman in to the Nebius registry using the nebius CLI credential
# helper for the active (or NEBIUS_PROFILE-selected) profile. Requires `jq`.
nebius-login:
	@echo "Logging in to $(NEBIUS_REGISTRY_HOST) via nebius credential helper..."
	@CRED="$$(echo $(NEBIUS_REGISTRY_HOST) | nebius $(NEBIUS_PROFILE_FLAG) registry docker-credential get)"; \
	printf '%s' "$$CRED" | jq -r .Secret | docker login $(NEBIUS_REGISTRY_HOST) \
	  --username "$$(printf '%s' "$$CRED" | jq -r .Username)" --password-stdin

# Build + tag robolab, then push it to the Nebius Container Registry instead of
# Docker Hub. Pair with `NEBIUS_IMAGE_REPO=$(NEBIUS_REGISTRY)/robolab` on
# workflows/nebius/eval.sh so the job pulls in-region.
push-robolab-cr: tag-robolab nebius-login
	@echo "Pushing $(IMAGE_NAME_ROBOLAB) to $(NEBIUS_REGISTRY)..."
	docker tag $(IMAGE_NAME_ROBOLAB):$(IMAGE_TAG) $(NEBIUS_REGISTRY)/robolab:$(IMAGE_TAG)
	docker tag $(IMAGE_NAME_ROBOLAB):$(GIT_SHA) $(NEBIUS_REGISTRY)/robolab:$(GIT_SHA)
	docker push $(NEBIUS_REGISTRY)/robolab:$(IMAGE_TAG)
	docker push $(NEBIUS_REGISTRY)/robolab:$(GIT_SHA)
	@echo "Pushed $(NEBIUS_REGISTRY)/robolab:{$(IMAGE_TAG),$(GIT_SHA)}"

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
	-docker rmi $(IMAGE_NAME_DREAMZERO):local
	-docker rmi $(IMAGE_NAME_DREAMZERO):$(IMAGE_TAG)
	-docker rmi $(IMAGE_NAME_DREAMZERO):$(GIT_SHA)
	-docker rmi $(TAG_DREAMZERO_BASE_LATEST)
	-docker rmi $(IMAGE_NAME_ROBOLAB):local
	-docker rmi $(IMAGE_NAME_ROBOLAB):$(IMAGE_TAG)
	-docker rmi $(IMAGE_NAME_ROBOLAB):$(GIT_SHA)
	@echo "All local images removed."

prune:
	@echo "Pruning dangling and unused Docker images..."
	docker image prune -f
	@echo "Prune complete."
