FROM runpod/worker-comfyui:5.8.5-base

# Install custom nodes for GGUF model loading and video output
RUN comfy-node-install ComfyUI-GGUF comfyui-videohelpersuite

# Add debug startup script with logging at every step
COPY debug_start.sh /debug_start.sh
RUN chmod +x /debug_start.sh

CMD ["/debug_start.sh"]
