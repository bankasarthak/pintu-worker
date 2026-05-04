FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/worker_requirements.txt
RUN pip install --no-cache-dir -r /workspace/worker_requirements.txt && \
    pip install --no-cache-dir numpy==1.26.4 && \
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY handler.py /workspace/
COPY inference.py /workspace/

ENV MODEL_DIR=/runpod-volume/models
ENV LORA_DIR=/runpod-volume/models/loras
ENV COMFYUI_DIR=/runpod-volume/ComfyUI
ENV COMFYUI_OUTPUT_DIR=/runpod-volume/ComfyUI/output
ENV COMFYUI_INPUT_DIR=/runpod-volume/ComfyUI/input
ENV LIGHTNING_CHECKPOINT=wan22_nsfw_lightning_q8l.gguf

CMD ["python", "-u", "/workspace/handler.py"]
