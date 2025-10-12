.PHONY: all build tag push clean prune help

# Image configuration
IMAGE_NAME := positronic-training

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

# Image tags
TAG_LATEST := $(REGISTRY_URL)/$(IMAGE_NAME):latest
TAG_VERSION := $(REGISTRY_URL)/$(IMAGE_NAME):v$(VERSION)
TAG_SHA := $(REGISTRY_URL)/$(IMAGE_NAME):$(GIT_SHA)
LOCAL_TAG := $(IMAGE_NAME):local

help:
	@echo "Positronic Training Container - Makefile"
	@echo ""
	@echo "Configuration:"
	@echo "  Image Name:     $(IMAGE_NAME)"
	@echo "  Version:        $(VERSION)"
	@echo "  Git SHA:        $(GIT_SHA)"
	@echo "  Registry FQDN:  $(NB_REGISTRY_FQDN)"
	@echo "  Registry Path:  $(NB_REGISTRY_PATH)"
	@echo "  Registry URL:   $(REGISTRY_URL)"
	@echo ""
	@echo "Targets:"
	@echo "  make build      Build the training image with baked source code"
	@echo "  make tag        Tag the image (depends on build)"
	@echo "  make push       Push all tags to Nebius (depends on tag)"
	@echo "  make all        Build, tag, and push (alias for push)"
	@echo "  make clean      Remove local training images"
	@echo "  make prune      Remove dangling/unused Docker images"
	@echo "  make help       Show this help message"

build:
	@echo "Building $(IMAGE_NAME) with source code baked in..."
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: Could not extract version from pyproject.toml"; \
		exit 1; \
	fi
	docker build -f Dockerfile.training.cloud -t $(LOCAL_TAG) ..

tag: build
	@echo "Tagging image with multiple tags..."
	@if [ -z "$(NB_REGISTRY_FQDN)" ] || [ -z "$(NB_REGISTRY_PATH)" ]; then \
		echo "Error: Nebius configuration not found. Please ensure:"; \
		echo "  1. Nebius CLI is configured (run: nebius config list)"; \
		echo "  2. Registry 'positronic' exists (run: nebius registry list)"; \
		exit 1; \
	fi
	docker tag $(LOCAL_TAG) $(TAG_LATEST)
	docker tag $(LOCAL_TAG) $(TAG_VERSION)
	docker tag $(LOCAL_TAG) $(TAG_SHA)
	@echo "Tagged with:"
	@echo "  - $(TAG_LATEST)"
	@echo "  - $(TAG_VERSION)"
	@echo "  - $(TAG_SHA)"

push: tag
	@echo "Pushing images to Nebius Container Registry..."
	docker push $(TAG_LATEST)
	docker push $(TAG_VERSION)
	docker push $(TAG_SHA)
	@echo ""
	@echo "Successfully pushed to Nebius Container Registry!"
	@echo "To use on cloud instances, set:"
	@echo "  export IMAGE_REGISTRY=$(REGISTRY_URL)/"

all: push

clean:
	@echo "Removing local training images..."
	-docker rmi $(LOCAL_TAG)
	-docker rmi $(TAG_LATEST)
	-docker rmi $(TAG_VERSION)
	-docker rmi $(TAG_SHA)
	@echo "Local images removed."

prune:
	@echo "Pruning dangling and unused Docker images..."
	docker image prune -f
	@echo "Prune complete."
