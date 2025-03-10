FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ADD --chmod=644 http://robotpkg.openrobots.org/packages/debian/robotpkg.gpg /robotpkg.gpg
RUN echo "deb [arch=amd64 signed-by=/robotpkg.gpg] http://robotpkg.openrobots.org/packages/debian/pub jammy robotpkg" \
 >> /etc/apt/sources.list.d/robotpkg.list

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    cmake \
    libpoco-dev \
    libpcre3-dev \
    portaudio19-dev \
    python3-dev \
    libeigen3-dev \
    robotpkg-py310-pinocchio \
    google-perftools \
    libgoogle-perftools-dev \
    libturbojpeg \
    python3-xlib \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy

RUN uv venv --python 3.11

# activate venv to be used in docker run
ENV VIRTUAL_ENV=/.venv/
ENV PATH=/$VIRTUAL_ENV/bin:$PATH

# Pinocchio ENV setup
ENV PATH=/opt/openrobots/bin:$PATH
ENV PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH

# Install libfranka
RUN git clone --single-branch --recurse-submodules --branch 0.13.4 https://github.com/frankaemika/libfranka
RUN mkdir libfranka/build && cd libfranka/build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. && cmake --build . && cmake --install .
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# allow to use ssh to clone franky
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# install franky
RUN --mount=type=ssh \
    git clone --branch version-0.9.1 --recursive --recurse-submodules git@github.com:Positronic-Robotics/franky.git

# TODO: improve franky installation to properly include pybind11
RUN uv pip install pybind11[global]==2.13.6 setuptools
RUN uv pip install franky/ --no-build-isolation

# Copy project files
COPY requirements-hardware.txt .
COPY pyproject.toml .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements-hardware.txt

COPY third_party/kortex_api-2.7.0.post5-py3-none-any.whl .
RUN uv pip install kortex_api-2.7.0.post5-py3-none-any.whl

WORKDIR /positronic

ENV PYTHONPATH=/positronic:$PYTHONPATH

# Default command
CMD ["bash"]
