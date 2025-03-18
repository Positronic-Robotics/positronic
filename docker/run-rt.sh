#!/bin/bash

# Docker run script for real-time features enabled

# Check if NVIDIA GPUs are available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_FLAGS="--gpus all"
else
    GPU_FLAGS=""
fi

xhost + local:docker

# Run with real-time scheduling priority
docker run ${GPU_FLAGS} --shm-size 128G --rm \
    --volume $PWD:/positronic \
    --network=host \
    --privileged \
    --ulimit rtprio=99 \
    --ulimit memlock=-1 \
    -e DISPLAY=$DISPLAY \
    -it positronic/positronic \
    "$@"
