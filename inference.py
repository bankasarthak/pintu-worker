"""ComfyUI-based inference for Wan 2.2 Lightning I2V with LoRA support.

Uses ComfyUI's headless API to execute workflows. The workflow JSON defines
the full pipeline: GGUF model load -> LoRA -> CLIP -> image input -> KSampler -> video encode.
"""

import json
import os
import time
import uuid

import boto3
import httpx
from botocore.config import Config as BotoConfig

COMFYUI_URL = "http://127.0.0.1:8188"


class ComfyUIClient:
    """Headless ComfyUI client that submits workflow JSONs and retrieves outputs."""

    def __init__(self):
        self.base_url = COMFYUI_URL
        self.client = httpx.Client(timeout=600)

    def queue_prompt(self, workflow: dict) -> str:
        resp = self.client.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow},
        )
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    def wait_for_completion(self, prompt_id: str, timeout: float = 300) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            resp = self.client.get(f"{self.base_url}/history/{prompt_id}")
            resp.raise_for_status()
            data = resp.json()
            if prompt_id in data:
                return data[prompt_id]
            time.sleep(2)
        raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {timeout}s")

    def get_output_path(self, history: dict) -> str | None:
        outputs = history.get("outputs", {})
        for node_id, node_output in outputs.items():
            if "gifs" in node_output:
                return node_output["gifs"][0]["filename"]
            if "videos" in node_output:
                return node_output["videos"][0]["filename"]
        return None


class LightningPipeline:
    def __init__(self, model_dir: str, lora_dir: str):
        self.model_dir = model_dir
        self.lora_dir = lora_dir
        self.comfy = ComfyUIClient()
        self.s3 = self._init_r2()
        self.bucket = os.environ.get("R2_BUCKET_NAME", "pintu-media")
        self.public_url = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
        self.comfyui_output_dir = os.environ.get(
            "COMFYUI_OUTPUT_DIR", "/workspace/ComfyUI/output"
        )

    @staticmethod
    def _init_r2():
        return boto3.client(
            "s3",
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            config=BotoConfig(signature_version="s3v4"),
            region_name="auto",
        )

    def _build_workflow(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        fps: int,
        guidance_scale: float,
        num_inference_steps: int,
        lora: str | None,
        lora_scale: float,
        checkpoint: str,
    ) -> dict:
        """Build a ComfyUI workflow JSON for Wan 2.2 Lightning I2V."""
        workflow = {
            "1": {
                "class_type": "UNETLoaderGGUF",
                "inputs": {
                    "unet_name": checkpoint,
                },
            },
            "2": {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "clip_name2": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "type": "wan",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "wan_2.1_vae.safetensors",
                },
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["2", 0],
                },
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["2", 0],
                },
            },
            "6": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": image_path,
                },
            },
            "7": {
                "class_type": "CLIPVisionLoader",
                "inputs": {
                    "clip_name": "clip_vision_h.safetensors",
                },
            },
            "8": {
                "class_type": "CLIPVisionEncode",
                "inputs": {
                    "clip_vision": ["7", 0],
                    "image": ["6", 0],
                },
            },
            "9": {
                "class_type": "WanImageToVideo",
                "inputs": {
                    "width": 832,
                    "height": 480,
                    "length": num_frames,
                    "batch_size": 1,
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "vae": ["3", 0],
                    "clip_vision_output": ["8", 0],
                },
            },
        }

        sampler_model_input = ["1", 0]

        if lora is not None:
            workflow["10"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "lora_name": lora,
                    "strength_model": lora_scale,
                    "model": ["1", 0],
                },
            }
            sampler_model_input = ["10", 0]

        workflow["11"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": int.from_bytes(os.urandom(4), "big"),
                "steps": num_inference_steps,
                "cfg": guidance_scale,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": sampler_model_input,
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["9", 0],
            },
        }

        workflow["12"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["11", 0],
                "vae": ["3", 0],
            },
        }

        workflow["13"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "pintu",
                "format": "video/h264-mp4",
                "images": ["12", 0],
            },
        }

        return workflow

    def _download_image(self, image_url: str) -> str:
        """Download input image to ComfyUI's input dir."""
        resp = httpx.get(image_url, timeout=30)
        resp.raise_for_status()
        filename = f"input_{uuid.uuid4().hex}.jpg"
        input_dir = os.environ.get(
            "COMFYUI_INPUT_DIR", "/workspace/ComfyUI/input"
        )
        os.makedirs(input_dir, exist_ok=True)
        path = os.path.join(input_dir, filename)
        with open(path, "wb") as f:
            f.write(resp.content)
        return filename

    def generate(
        self,
        image_url: str,
        prompt: str,
        negative_prompt: str = "blurry, distorted, low quality",
        num_frames: int = 48,
        fps: int = 16,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 6,
        lora: str | None = None,
        lora_scale: float = 0.8,
        base_model: str = "wan22",
    ) -> str:
        checkpoint = os.environ.get(
            "LIGHTNING_CHECKPOINT",
            "wan22_nsfw_lightning_q8l.gguf",
        )

        image_filename = self._download_image(image_url)

        workflow = self._build_workflow(
            image_path=image_filename,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            fps=fps,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            lora=lora,
            lora_scale=lora_scale,
            checkpoint=checkpoint,
        )

        prompt_id = self.comfy.queue_prompt(workflow)
        history = self.comfy.wait_for_completion(prompt_id)
        output_filename = self.comfy.get_output_path(history)

        if output_filename is None:
            raise RuntimeError("ComfyUI produced no video output")

        output_path = os.path.join(self.comfyui_output_dir, output_filename)
        video_url = self._upload_video(output_path)

        os.unlink(output_path)
        return video_url

    def _upload_video(self, local_path: str) -> str:
        key = f"outputs/{uuid.uuid4().hex}.mp4"
        self.s3.upload_file(
            local_path,
            self.bucket,
            key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
        return f"{self.public_url}/{key}"
