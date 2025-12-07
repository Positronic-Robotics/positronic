.PHONY: all build tag push clean prune help build-base build-training tag-base tag-training push-base push-training

# Image configuration
IMAGE_NAME_TRAINING := positronic-training
IMAGE_NAME_BASE := positronic-base
IMAGE_NAME_OPENPI := positronic-internal-openpi
IMAGE_NAME_GROOT := positronic-internal-groot

# Extract version from pyproject.toml (first literal version entry)
VERSION := $(shell sed -n 's/^version = "\([^"]*\)"/\1/p' ../pyproject.toml | head -n 1)

# Get git commit SHA (short)
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Get Nebius configuration from CLI
# Extract registry info from JSON format
REGISTRY_INFO := $(shell nebius registry list --format json 2>/dev/null | jq -r '.items[] | select(.metadata.name=="positronic") | .metadata.id + " " + .status.registry_fqdn' || echo "")
NB_REGISTRY_PATH := $(shell echo "$(REGISTRY_INFO)" | cut -d' ' -f1 | cut -d- -f2)
NB_REGISTRY_FQDN := $(shell echo "$(REGISTRY_INFO)" | cut -d' ' -f2)

# Full registry URL
REGISTRY_URL := $(NB_REGISTRY_FQDN)/$(NB_REGISTRY_PATH)

# Image tags for base
TAG_BASE_LATEST := $(REGISTRY_URL)/$(IMAGE_NAME_BASE):latest
TAG_BASE_VERSION := $(REGISTRY_URL)/$(IMAGE_NAME_BASE):v$(VERSION)
TAG_BASE_SHA := $(REGISTRY_URL)/$(IMAGE_NAME_BASE):$(GIT_SHA)

# Image tags for training
TAG_TRAINING_LATEST := $(REGISTRY_URL)/$(IMAGE_NAME_TRAINING):latest
TAG_TRAINING_VERSION := $(REGISTRY_URL)/$(IMAGE_NAME_TRAINING):v$(VERSION)
TAG_TRAINING_SHA := $(REGISTRY_URL)/$(IMAGE_NAME_TRAINING):$(GIT_SHA)
LOCAL_TAG_TRAINING := $(IMAGE_NAME_TRAINING):local
TAG_OPENPI_LATEST := $(REGISTRY_URL)/$(IMAGE_NAME_OPENPI):latest
TAG_GROOT_LATEST := $(REGISTRY_URL)/$(IMAGE_NAME_GROOT):latest


# Local base image tag
LOCAL_TAG_BASE := $(IMAGE_NAME_BASE):local

help:
	@echo "Positronic Docker Build System - Makefile"
	@echo ""
	@echo "Configuration:"
	@echo "  Training Image: $(IMAGE_NAME_TRAINING)"
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
	@echo "  make build           Build base and training images"
	@echo ""
	@echo "  make tag-base        Tag the base image (depends on build-base)"
	@echo "  make tag-training    Tag the training image (depends on build-training)"
	@echo "  make tag             Tag both images"
	@echo ""
	@echo "  make push-base       Push base tags to Nebius (depends on tag-base)"
	@echo "  make push-training   Push training tags to Nebius (depends on tag-training)"
	@echo "  make push            Push all tags to Nebius (depends on tag)"
	@echo ""
	@echo "  make all             Build, tag, and push all images (alias for push)"
	@echo "  make clean           Remove all local images"
	@echo "  make prune           Remove dangling/unused Docker images"
	@echo "  make help            Show this help message"

build-base:
	@echo "Building base image with common dependencies..."
	docker build --pull --platform linux/amd64 -f Dockerfile.base -t $(LOCAL_TAG_BASE) ..

build-training: build-base
	@echo "Building $(IMAGE_NAME_TRAINING) with source code baked in..."
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: Could not extract version from pyproject.toml"; \
		exit 1; \
	fi
	docker build --platform linux/amd64 -f Dockerfile.training.cloud -t $(LOCAL_TAG_TRAINING) ..

build-openpi: build-training
	@echo "Building $(IMAGE_NAME_OPENPI)..."
	docker build --platform linux/amd64 --build-arg BASE_IMAGE=$(LOCAL_TAG_TRAINING) -f Dockerfile.openpi -t $(IMAGE_NAME_OPENPI):local ../..

build-groot: build-training
	@echo "Building $(IMAGE_NAME_GROOT)..."
	docker build --pull --platform linux/amd64 --build-arg BASE_IMAGE=$(REGISTRY_URL)/positronic-groot:latest -f Dockerfile.groot -t $(IMAGE_NAME_GROOT):local ..

build: build-base build-training

tag-base: build-base
	@echo "Tagging base image with multiple tags..."
	@if [ -z "$(NB_REGISTRY_FQDN)" ] || [ -z "$(NB_REGISTRY_PATH)" ]; then \
		echo "Error: Nebius configuration not found. Please ensure:"; \
		echo "  1. Nebius CLI is configured (run: nebius config list)"; \
		echo "  2. Registry 'positronic' exists (run: nebius registry list)"; \
		exit 1; \
	fi
	docker tag $(LOCAL_TAG_BASE) $(TAG_BASE_LATEST)
	docker tag $(LOCAL_TAG_BASE) $(TAG_BASE_VERSION)
	docker tag $(LOCAL_TAG_BASE) $(TAG_BASE_SHA)
	@echo "Tagged with:"
	@echo "  - $(TAG_BASE_LATEST)"
	@echo "  - $(TAG_BASE_VERSION)"
	@echo "  - $(TAG_BASE_SHA)"

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

tag: tag-base tag-training

push-base: tag-base
	@echo "Pushing base images to Nebius Container Registry..."
	docker push $(TAG_BASE_LATEST)
	docker push $(TAG_BASE_VERSION)
	docker push $(TAG_BASE_SHA)
	@echo ""
	@echo "Successfully pushed base images to Nebius Container Registry!"
	@echo "To use on cloud instances, set:"
	@echo "  export IMAGE_REGISTRY=$(REGISTRY_URL)/"

push-training: tag-training
	@echo "Pushing training images to Nebius Container Registry..."
	docker push $(TAG_TRAINING_LATEST)
	docker push $(TAG_TRAINING_VERSION)
	docker push $(TAG_TRAINING_SHA)
	@echo ""
	@echo "Successfully pushed training images to Nebius Container Registry!"
	@echo "To use on cloud instances, set:"
	@echo "  export IMAGE_REGISTRY=$(REGISTRY_URL)/"

push: push-base push-training push-openpi push-groot

tag-openpi: build-openpi
	@echo "Tagging openpi image..."
	docker tag $(IMAGE_NAME_OPENPI):local $(TAG_OPENPI_LATEST)
	@echo "Tagged with: $(TAG_OPENPI_LATEST)"

push-openpi: tag-openpi
	@echo "Pushing $(IMAGE_NAME_OPENPI) to Nebius..."
	docker push $(TAG_OPENPI_LATEST)

tag-groot: build-groot
	@echo "Tagging groot image..."
	docker tag $(IMAGE_NAME_GROOT):local $(TAG_GROOT_LATEST)
	@echo "Tagged with: $(TAG_GROOT_LATEST)"

push-groot: tag-groot
	@echo "Pushing $(IMAGE_NAME_GROOT) to Nebius..."
	docker push $(TAG_GROOT_LATEST)

all: push

clean:
	@echo "Removing all local images..."
	-docker rmi $(LOCAL_TAG_BASE)
	-docker rmi $(TAG_BASE_LATEST)
	-docker rmi $(TAG_BASE_VERSION)
	-docker rmi $(TAG_BASE_SHA)
	-docker rmi $(LOCAL_TAG_TRAINING)
	-docker rmi $(TAG_TRAINING_LATEST)
	-docker rmi $(TAG_TRAINING_VERSION)
	-docker rmi $(TAG_TRAINING_SHA)
	@echo "All local images removed."

prune:
	@echo "Pruning dangling and unused Docker images..."
	docker image prune -f
	@echo "Prune complete."
