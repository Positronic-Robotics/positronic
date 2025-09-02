FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    cmake \
    libpoco-dev \
    libpcre3-dev \
    portaudio19-dev \
    libeigen3-dev \
    libturbojpeg \
    libfmt-dev \
    lsb-release \
    libhidapi-dev  # Spacemouse

RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | tee /etc/apt/keyrings/robotpkg.asc
RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
 | tee /etc/apt/sources.list.d/robotpkg.list

RUN apt-get update
RUN apt-get install -y robotpkg-pinocchio

# Ensure CMake and runtime can find robotpkg installs (e.g., Pinocchio)
ENV ROBOTPKG_PREFIX=/opt/openrobots
ENV CMAKE_PREFIX_PATH=${ROBOTPKG_PREFIX}
ENV LD_LIBRARY_PATH=${ROBOTPKG_PREFIX}/lib:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=${ROBOTPKG_PREFIX}/lib/pkgconfig

ENV UV_LINK_MODE=copy
RUN uv venv --python 3.11

# activate venv to be used in docker run
ENV VIRTUAL_ENV=/.venv/
ENV PATH=/$VIRTUAL_ENV/bin:$PATH

# Install libfranka
RUN git clone --single-branch --recurse-submodules --branch 0.15.3 https://github.com/frankaemika/libfranka
RUN mkdir libfranka/build && cd libfranka/build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. && cmake --build . && cmake --install .
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Copy project metadata and lock for reproducible deps
COPY pyproject.toml uv.lock .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra hardware --no-install-project

WORKDIR /positronic

ENV PYTHONPATH=/positronic
# Default command
CMD ["bash"]
