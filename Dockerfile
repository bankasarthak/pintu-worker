FROM runpod/worker-comfyui:5.8.5-base

# System deps needed by custom nodes (libglib for GGUF, ffmpeg for VideoHelperSuite)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libglib2.0-0 \
       libsm6 \
       libxrender1 \
       libxext6 \
       ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install custom nodes for GGUF model loading and video output
RUN comfy-node-install ComfyUI-GGUF comfyui-videohelpersuite
