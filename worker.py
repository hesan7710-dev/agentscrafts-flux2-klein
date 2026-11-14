# FLUX.2 Klein RunPod Serverless worker
# Model inference is loaded once per worker process.
from __future__ import annotations
import base64, io, json, os, time, uuid
from pathlib import Path
from typing import Any

import runpod
import torch
from PIL import Image, ImageOps
from diffusers import Flux2KleinPipeline

MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.2-klein-base-4B")
PROMPTS_PATH = Path(os.getenv("PROMPTS_PATH", "/app/prompts.json"))
MAX_INPUT_BYTES = int(os.getenv("MAX_INPUT_BYTES", str(20 * 1024 * 1024)))
pipe = None
prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def load_model():
    global pipe
    if pipe is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for FLUX.2 Klein inference.")
        pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        # Offload keeps memory requirements lower; model stays resident on CPU between calls.
        pipe.enable_model_cpu_offload()
    return pipe


def decode_image(value: str) -> Image.Image:
    if not isinstance(value, str) or not value:
        raise ValueError("input.image_base64 is required.")
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    raw = base64.b64decode(value, validate=True)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError(f"Input image exceeds {MAX_INPUT_BYTES} bytes.")
    image = Image.open(io.BytesIO(raw))
    image.load()
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width * image.height > 40_000_000:
        raise ValueError("Image dimensions exceed the 40-megapixel safety limit.")
    return image


def store_result(raw: bytes, job_id: str) -> dict[str, Any]:
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return {"image_base64": base64.b64encode(raw).decode("ascii"), "content_type": "image/png"}
    import boto3
    client = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        region_name=os.getenv("S3_REGION", "auto"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    )
    key = f"{os.getenv('S3_PREFIX', 'agentscrafts/flux')}/{job_id}.png"
    client.put_object(Bucket=bucket, Key=key, Body=raw, ContentType="image/png")
    expires = int(os.getenv("S3_PRESIGNED_URL_SECONDS", "3600"))
    url = client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)
    return {"image_url": url, "storage_key": key, "content_type": "image/png"}


def handler(job):
    data = job.get("input") or {}
    started = time.time()
    job_id = str(job.get("id") or uuid.uuid4().hex)
    try:
        image = decode_image(data.get("image_base64"))
        prompt_type = data.get("prompt_type", "general")
        if prompt_type not in prompts:
            raise ValueError(f"Unknown prompt_type. Allowed: {', '.join(prompts.keys())}")
        if data.get("full_reconstruction"):
            # The authoritative V2 prompt JSON intentionally contains no reconstruction suffix.
            # Do not invent one; caller must supply the approved suffix explicitly.
            suffix = data.get("reconstruction_suffix")
            if not isinstance(suffix, str) or not suffix.strip():
                raise ValueError("full_reconstruction requires the approved reconstruction_suffix; none is bundled in V2 prompt library.")
        else:
            suffix = ""
        custom_prompt = data.get("custom_prompt")
        prompt = custom_prompt.strip() if isinstance(custom_prompt, str) and custom_prompt.strip() else prompts[prompt_type]["text"]
        if suffix:
            prompt = prompt.rstrip() + "\n\n" + suffix.strip()
        generator = None
        seed = data.get("seed")
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
        model = load_model()
        result = model(image=image, prompt=prompt, generator=generator).images[0]
        output = io.BytesIO()
        result.save(output, format="PNG")
        stored = store_result(output.getvalue(), job_id)
        return {"status": "completed", "job_id": job_id, "prompt_type": prompt_type,
                "seed": seed, "width": result.width, "height": result.height,
                "elapsed_seconds": round(time.time() - started, 2), **stored}
    except Exception as exc:
        return {"status": "failed", "job_id": job_id, "error": str(exc),
                "error_type": type(exc).__name__}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
