# Base image for DreamZero inference and training.
# Produces: positro/dreamzero-base:latest
#
# Build (from repo root):
#   docker build -t positro/dreamzero-base:latest -f positronic/vendors/dreamzero/Dockerfile .
#
# Dockerfile.serve-droid adds the standalone server; Dockerfile.training adds the training stack.

FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (GL libs needed for opencv/image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create Python 3.11 venv
RUN uv python install 3.11 && uv venv /.venv --python 3.11
ENV VIRTUAL_ENV=/.venv
ENV PATH="/.venv/bin:$PATH"

# Heavy compilation layers — placed before git clone so they stay cached across SHA updates.
# Install PyTorch 2.8 with CUDA 12.9 first (dreamzero depends on it)
RUN uv pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129

# Install flash-attn and transformer-engine (need torch + CUDA at build time)
RUN uv pip install packaging wheel
RUN uv pip install flash-attn --no-build-isolation
RUN uv pip install "transformer-engine[pytorch]" --no-build-isolation

# Clone DreamZero at pinned commit
ARG DREAMZERO_SHA=d70a8025ac77f7486f38032c718a1ca814a4d8c7
RUN git clone https://github.com/dreamzero0/dreamzero.git /dreamzero && \
    cd /dreamzero && git checkout ${DREAMZERO_SHA}

# Install DreamZero and all its dependencies
RUN cd /dreamzero && uv pip install -e .

# Patch cudnn_attention.py for transformer-engine >= 2.12 compatibility.
#
# Background: DreamZero's cudnn_attention.py calls transformer_engine_torch.fused_attn_fwd()
# and fused_attn_bwd() directly (the C++ bindings), building the positional argument tuple
# manually with version guards for TE 2.8, 2.9, and 2.10. DreamZero doesn't pin TE, so
# `uv pip install "transformer-engine[pytorch]"` pulls the latest (2.12 at time of writing).
#
# TE 2.12 added a new bool parameter `bottom_right_diagonal` to both fused_attn_fwd and
# fused_attn_bwd. It controls causal mask alignment and sits between the `window_size` and
# `cu_seqlens_q` args (fwd) / between `window_size` and `deterministic` (bwd). Without this
# arg the C++ binding raises TypeError because tensors land in a bool slot.
#
# The patch inserts `if _TE_VER >= (2, 12): args += (False,)` in both the fwd and bwd
# functions, matching the existing version guard style (see _TE_VER >= (2, 9) etc.).
RUN python -c "\
f='/dreamzero/groot/vla/model/dreamzero/modules/cudnn_attention.py'; \
t=open(f).read(); \
t=t.replace( \
  '    args += (\n        tuple(window_size),\n        cu_seqlens_q,', \
  '    args += (\n        tuple(window_size),\n    )\n    if _TE_VER >= (2, 12):\n        args += (False,)  # bottom_right_diagonal\n    args += (\n        cu_seqlens_q,', 1); \
t=t.replace( \
  '    args += (\n        window_size,\n        deterministic,\n        cu_seqlens_q,', \
  '    args += (\n        window_size,\n    )\n    if _TE_VER >= (2, 12):\n        args += (False,)  # bottom_right_diagonal\n    args += (\n        deterministic,\n        cu_seqlens_q,', 1); \
open(f,'w').write(t)"

WORKDIR /dreamzero
