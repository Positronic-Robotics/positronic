FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    libpoco-dev \
    libpcre3-dev \
    portaudio19-dev \
    libeigen3-dev \
    libturbojpeg \
    libhidapi-dev  # Spacemouse

ENV UV_LINK_MODE=copy

RUN uv venv --python 3.11

# activate venv to be used in docker run
ENV VIRTUAL_ENV=/.venv/
ENV PATH=/$VIRTUAL_ENV/bin:$PATH


# Install libfranka
RUN git clone --single-branch --recurse-submodules --branch 0.13.4 https://github.com/frankaemika/libfranka
RUN mkdir libfranka/build && cd libfranka/build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. && cmake --build . && cmake --install .
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Copy project metadata and lock for reproducible deps
COPY pyproject.toml uv.lock .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra hardware --no-install-project

WORKDIR /positronic

ENV PYTHONPATH=/positronic:/positronic/lerobot
# Default command
CMD ["bash"]
