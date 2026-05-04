"""RunPod Serverless handler for Wan 2.2 Lightning I2V via ComfyUI.

Starts ComfyUI in headless mode from the Network Volume, then processes
jobs by building and submitting workflow JSONs.
"""

import os
import subprocess
import time

import runpod

from inference import LightningPipeline

pipeline: LightningPipeline | None = None
comfyui_process: subprocess.Popen | None = None


def setup_symlinks():
    """Recreate model symlinks from volume into ComfyUI directories."""
    comfyui_dir = os.environ.get("COMFYUI_DIR", "/runpod-volume/ComfyUI")
    model_dir = os.environ.get("MODEL_DIR", "/runpod-volume/models")

    dirs_map = {
        "checkpoints": "checkpoints",
        "loras": "loras",
        "clip": "text_encoders",
        "clip_vision": "clip_vision",
        "vae": "vae",
    }

    for src_name, dst_name in dirs_map.items():
        src = os.path.join(model_dir, src_name)
        dst = os.path.join(comfyui_dir, "models", dst_name)
        os.makedirs(dst, exist_ok=True)
        if os.path.isdir(src):
            for f in os.listdir(src):
                if f.startswith("."):
                    continue
                src_file = os.path.join(src, f)
                dst_file = os.path.join(dst, f)
                if os.path.isfile(src_file) and not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)

    # GGUF checkpoints also go in diffusion_models
    diff_dir = os.path.join(comfyui_dir, "models", "diffusion_models")
    os.makedirs(diff_dir, exist_ok=True)
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.endswith(".gguf"):
                src_file = os.path.join(ckpt_dir, f)
                dst_file = os.path.join(diff_dir, f)
                if not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)


def start_comfyui():
    """Launch ComfyUI server in the background."""
    global comfyui_process
    if comfyui_process is not None:
        return

    comfyui_dir = os.environ.get("COMFYUI_DIR", "/runpod-volume/ComfyUI")

    if not os.path.isdir(comfyui_dir):
        raise RuntimeError(
            f"ComfyUI not found at {comfyui_dir}. "
            "Make sure the Network Volume is mounted."
        )

    setup_symlinks()

    comfyui_process = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"],
        cwd=comfyui_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    import httpx
    for _ in range(90):
        try:
            resp = httpx.get("http://127.0.0.1:8188/system_stats", timeout=5)
            if resp.status_code == 200:
                print("ComfyUI server is ready")
                return
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError("ComfyUI failed to start within 180 seconds")


def get_pipeline() -> LightningPipeline:
    global pipeline
    if pipeline is None:
        start_comfyui()
        model_dir = os.environ.get("MODEL_DIR", "/runpod-volume/models")
        lora_dir = os.environ.get("LORA_DIR", "/runpod-volume/models/loras")
        pipeline = LightningPipeline(model_dir, lora_dir)
    return pipeline


def handler(job: dict) -> dict:
    job_input = job["input"]
    pipe = get_pipeline()

    video_url = pipe.generate(
        image_url=job_input["image_url"],
        prompt=job_input["prompt"],
        negative_prompt=job_input.get("negative_prompt", "blurry, distorted, low quality"),
        num_frames=job_input.get("num_frames", 48),
        fps=job_input.get("fps", 16),
        guidance_scale=job_input.get("guidance_scale", 5.0),
        num_inference_steps=job_input.get("num_inference_steps", 6),
        lora=job_input.get("lora"),
        lora_scale=job_input.get("lora_scale", 0.8),
        base_model=job_input.get("base_model", "wan22"),
    )

    return {"video_url": video_url}


runpod.serverless.start({"handler": handler})
