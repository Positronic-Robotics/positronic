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
    && rm -rf /var/lib/apt/lists/*


# Copy project files
COPY requirements-all.txt .
COPY pyproject.toml .

ENV UV_LINK_MODE=copy
ENV VIRTUAL_ENV=/.venv/
ENV PATH=/$VIRTUAL_ENV/bin:$PATH

RUN uv venv --python 3.11
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements-all.txt


# Pinocchio ENV setup
ENV PATH=/opt/openrobots/bin:$PATH
ENV PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
#ENV PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH
ENV CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH

# Install libfranka
RUN git clone --single-branch --recurse-submodules --branch 0.13.4 https://github.com/frankaemika/libfranka
RUN mkdir libfranka/build && cd libfranka/build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. && cmake --build . && cmake --install .
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH


# install franky
RUN git clone --branch v0.9.1 https://github.com/TimSchneider42/franky.git
RUN cd franky && git submodule update --init --recursive
RUN uv pip install pybind11[global]
RUN uv pip install franky/ --no-build-isolation

WORKDIR /positronic

# Default command
CMD ["bash"]