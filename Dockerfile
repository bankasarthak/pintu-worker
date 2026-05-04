FROM runpod/worker-comfyui:5.8.5-base

# Install custom nodes for GGUF model loading and video output
RUN comfy-node-install ComfyUI-GGUF comfyui-videohelpersuite
