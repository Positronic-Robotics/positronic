#!/bin/bash

# Check if NVIDIA GPUs are available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_FLAGS="--gpus all"
else
    GPU_FLAGS=""
fi

xhost + local:docker

docker run ${GPU_FLAGS} --shm-size 128G --rm \
    --volume $PWD:/positronic \
    --volume $PWD/../lerobot:/lerobot \
    --volume /dev/v4l/:/dev/v4l/ \
    --volume $PWD/.cache:/root/.cache/ \
    --network=host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -it positronic/positronic "$@"
