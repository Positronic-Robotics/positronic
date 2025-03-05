xhost + local:docker

docker run --gpus all --shm-size 128G --rm \
    --volume $PWD:/positronic \
    --network=host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -it positronic/positronic "$@"
