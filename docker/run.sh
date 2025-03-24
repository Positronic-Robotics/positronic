xhost + local:docker

docker run --gpus all --shm-size 128G --rm \
    --volume $PWD:/positronic \
    --volume $PWD/../lerobot:/lerobot \
    --network=host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -it positronic/positronic "$@"
