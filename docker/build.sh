docker build -t positronic/positronic --file docker/Dockerfile --build-arg LIBFRANKA_VERSION="${LIBFRANKA_VERSION:-0.15.3}" \
    --ssh default .
