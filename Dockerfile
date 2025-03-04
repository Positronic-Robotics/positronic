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
COPY requirements.txt .
COPY pyproject.toml .
COPY uv.lock .

ENV UV_LINK_MODE=copy

RUN uv venv --python 3.11
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements-all.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /
# Default command
CMD ["bash"]