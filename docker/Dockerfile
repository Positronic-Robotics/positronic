FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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
    && rm -rf /var/lib/apt/lists/*


# Copy project files
COPY requirements-all.txt .
COPY pyproject.toml .

ENV UV_LINK_MODE=copy

RUN uv venv --python 3.11
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements-all.txt
RUN . .venv/bin/activate

WORKDIR /positronic
# Default command
CMD ["bash"]