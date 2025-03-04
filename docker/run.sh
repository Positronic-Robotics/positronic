docker run --gpus all --shm-size 128G --rm \
    --volume $PWD:/positronic \
    --network=host \
    --privileged \
    -it positronic/positronic "$@"