#!/bin/bash
xhost + local:docker

# Run with real-time scheduling priority
docker run --shm-size 128G --rm \
    --volume $PWD:/positronic \
    --network=host \
    --privileged \
    --ulimit rtprio=99 \
    --ulimit memlock=-1 \
    -e DISPLAY=$DISPLAY \
    -it positronic/positronic \
    chrt -f 99 "$@"
