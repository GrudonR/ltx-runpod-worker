import base64
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import runpod
from huggingface_hub import hf_hub_download, snapshot_download


LOGGER = logging.getLogger("ltx-runpod-official")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

LTX_REPO_DIR = Path(os.getenv("LTX_REPO_DIR", "/opt/LTX-2"))
LTX_VENV_PYTHON = LTX_REPO_DIR / ".venv" / "bin" / "python"
LTX_REPO_COMMIT = os.getenv("LTX_REPO_COMMIT", "unknown")
LTX_MODEL_REPO = os.getenv("LTX_MODEL_REPO", "Lightricks/LTX-2.3")
LTX_GEMMA_REPO = os.getenv("LTX_GEMMA_REPO", "google/gemma-3-12b-it-qat-q4_0-unquantized")
LTX_CHECKPOINT_NAME = os.getenv("LTX_CHECKPOINT_NAME", "ltx-2.3-22b-distilled.safetensors")
LTX_CHECKPOINT_FULL_NAME = os.getenv("LTX_CHECKPOINT_FULL_NAME", "ltx-2.3-22b-dev.safetensors")
LTX_DISTILLED_LORA_NAME = os.getenv("LTX_DISTILLED_LORA_NAME", "ltx-2.3-22b-distilled-lora-384.safetensors")
LTX_SPATIAL_UPSAMPLER_NAME = os.getenv("LTX_SPATIAL_UPSAMPLER_NAME", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

VALID_PIPELINES = {"distilled", "two_stages", "two_stages_hq"}
LTX_CACHE_ROOT = Path(os.getenv("LTX_CACHE_ROOT", "/runpod-volume/ltx"))
LTX_WORK_ROOT = Path(os.getenv("LTX_WORK_ROOT", "/tmp/ltx"))
LOCAL_OUTPUT_DIR = Path(os.getenv("LOCAL_OUTPUT_DIR", "/runpod-volume/outputs"))
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
MAX_INLINE_OUTPUT_MB = float(os.getenv("MAX_INLINE_OUTPUT_MB", "50"))
DEFAULT_FRAME_RATE = 24.0
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 576
DEFAULT_NUM_FRAMES = 121
DEFAULT_SEED = 10


_ASSET_CACHE: dict[str, str] | None = None


class InputError(ValueError):
    pass


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_hf_token() -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required for official LTX 2.3 model access.")
    return HF_TOKEN


def download_file(url: str, dest: Path) -> None:
    """Download a file from a URL to a local path."""
    request = urllib.request.Request(url, headers={"User-Agent": "ltx-runpod-worker/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with dest.open("wb") as f:
            shutil.copyfileobj(response, f)


def resolve_image_input(image_spec: dict[str, Any], work_dir: Path, index: int) -> dict[str, Any]:
    """Resolve an image input to a local file path.

    Accepts either:
      - {"url": "https://..."} — downloaded to work_dir
      - {"base64": "..."} — decoded to work_dir
    Plus required: frame_index (int), strength (float)
    Optional: crf (int)
    """
    frame_index = image_spec.get("frame_index")
    if frame_index is None:
        raise InputError(f"images[{index}]: frame_index is required")
    frame_index = int(frame_index)

    strength = float(image_spec.get("strength", 1.0))
    if not 0.0 <= strength <= 1.0:
        raise InputError(f"images[{index}]: strength must be between 0.0 and 1.0")

    crf = image_spec.get("crf")

    # Determine file extension from URL or default to .png
    ext = ".png"
    url = image_spec.get("url")
    b64 = image_spec.get("base64")

    if url and b64:
        raise InputError(f"images[{index}]: provide either url or base64, not both")
    if not url and not b64:
        raise InputError(f"images[{index}]: provide either url or base64")

    dest = work_dir / f"cond_image_{index}{ext}"

    if url:
        download_file(url, dest)
    else:
        dest.write_bytes(base64.b64decode(b64))

    result = {
        "path": str(dest),
        "frame_index": frame_index,
        "strength": strength,
    }
    if crf is not None:
        result["crf"] = int(crf)
    return result


def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise InputError("prompt is required")

    width = int(payload.get("width", DEFAULT_WIDTH))
    height = int(payload.get("height", DEFAULT_HEIGHT))
    num_frames = int(payload.get("num_frames", DEFAULT_NUM_FRAMES))
    frame_rate = float(payload.get("frame_rate", DEFAULT_FRAME_RATE))
    seed = int(payload.get("seed", DEFAULT_SEED))
    quantization = payload.get("quantization")
    if quantization is not None:
        quantization = str(quantization).strip()
        if quantization not in {"fp8-cast", "fp8-scaled-mm"}:
            raise InputError("quantization must be one of: fp8-cast, fp8-scaled-mm")

    if width <= 0 or height <= 0:
        raise InputError("width and height must be positive")
    if width % 64 != 0 or height % 64 != 0:
        raise InputError("official two-stage distilled pipeline requires width and height divisible by 64")
    if num_frames <= 0 or (num_frames - 1) % 8 != 0:
        raise InputError("official LTX 2.3 pipelines require num_frames = (8 * k) + 1")
    if frame_rate <= 0:
        raise InputError("frame_rate must be positive")

    # Validate image conditioning specs (resolved later when work_dir exists)
    raw_images = payload.get("images") or []
    if not isinstance(raw_images, list):
        raise InputError("images must be a list")
    for i, img in enumerate(raw_images):
        if not isinstance(img, dict):
            raise InputError(f"images[{i}]: must be an object with url/base64, frame_index, strength")
        if "frame_index" not in img:
            raise InputError(f"images[{i}]: frame_index is required")

    pipeline = str(payload.get("pipeline", "distilled")).strip()
    if pipeline not in VALID_PIPELINES:
        raise InputError(f"pipeline must be one of: {', '.join(sorted(VALID_PIPELINES))}")

    return {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "seed": seed,
        "quantization": quantization,
        "pipeline": pipeline,
        "enhance_prompt": bool(payload.get("enhance_prompt", False)),
        "output_upload_url": payload.get("output_upload_url"),
        "return_base64": bool(payload.get("return_base64", True)),
        "raw_images": raw_images,
    }


def ensure_assets() -> dict[str, str]:
    global _ASSET_CACHE
    if _ASSET_CACHE is not None:
        return _ASSET_CACHE

    token = require_hf_token()
    model_dir = LTX_CACHE_ROOT / "models" / "Lightricks-LTX-2.3"
    gemma_dir = LTX_CACHE_ROOT / "models" / "google-gemma-3-12b-it-qat-q4_0-unquantized"
    ensure_directory(model_dir)
    ensure_directory(gemma_dir)
    ensure_directory(LTX_WORK_ROOT)
    ensure_directory(LOCAL_OUTPUT_DIR)

    checkpoint_path = hf_hub_download(
        repo_id=LTX_MODEL_REPO,
        filename=LTX_CHECKPOINT_NAME,
        token=token,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    upsampler_path = hf_hub_download(
        repo_id=LTX_MODEL_REPO,
        filename=LTX_SPATIAL_UPSAMPLER_NAME,
        token=token,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id=LTX_GEMMA_REPO,
        token=token,
        local_dir=str(gemma_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.json",
            "*.model",
            "*.safetensors",
        ],
    )

    # Full (non-distilled) checkpoint + distilled LoRA for two-stage pipelines
    full_checkpoint_path = hf_hub_download(
        repo_id=LTX_MODEL_REPO,
        filename=LTX_CHECKPOINT_FULL_NAME,
        token=token,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    distilled_lora_path = hf_hub_download(
        repo_id=LTX_MODEL_REPO,
        filename=LTX_DISTILLED_LORA_NAME,
        token=token,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )

    _ASSET_CACHE = {
        "checkpoint_path": checkpoint_path,
        "full_checkpoint_path": full_checkpoint_path,
        "distilled_lora_path": distilled_lora_path,
        "spatial_upsampler_path": upsampler_path,
        "gemma_root": str(gemma_dir),
    }
    return _ASSET_CACHE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maybe_upload_file(path: Path, upload_url: str | None) -> str | None:
    if not upload_url:
        return None

    with path.open("rb") as handle:
        data = handle.read()

    request = urllib.request.Request(
        upload_url,
        data=data,
        method="PUT",
        headers={
            "Content-Type": "video/mp4",
            "Content-Length": str(len(data)),
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status not in {200, 201}:
            raise RuntimeError(f"upload failed with HTTP {response.status}")
    return upload_url


def maybe_inline_output(path: Path, return_base64: bool) -> str | None:
    if not return_base64:
        return None

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_INLINE_OUTPUT_MB:
        raise InputError(
            f"output is {size_mb:.2f} MB, larger than MAX_INLINE_OUTPUT_MB={MAX_INLINE_OUTPUT_MB:g}; "
            "use output_upload_url instead"
        )

    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_command(
    assets: dict[str, str],
    payload: dict[str, Any],
    output_path: Path,
    resolved_images: list[dict[str, Any]],
) -> list[str]:
    pipeline = payload.get("pipeline", "distilled")

    # Map pipeline name to module and checkpoint args
    if pipeline == "distilled":
        module = "ltx_pipelines.distilled"
        command = [
            str(LTX_VENV_PYTHON), "-m", module,
            "--distilled-checkpoint-path", assets["checkpoint_path"],
            "--spatial-upsampler-path", assets["spatial_upsampler_path"],
        ]
    elif pipeline in ("two_stages", "two_stages_hq"):
        module = "ltx_pipelines.ti2vid_two_stages_hq" if pipeline == "two_stages_hq" else "ltx_pipelines.ti2vid_two_stages"
        command = [
            str(LTX_VENV_PYTHON), "-m", module,
            "--checkpoint-path", assets["full_checkpoint_path"],
            "--spatial-upsampler-path", assets["spatial_upsampler_path"],
            "--distilled-lora", assets["distilled_lora_path"], "0.8",
        ]
    else:
        raise InputError(f"unknown pipeline: {pipeline}")

    # Common args
    command.extend([
        "--gemma-root", assets["gemma_root"],
        "--prompt", payload["prompt"],
        "--output-path", str(output_path),
        "--seed", str(payload["seed"]),
        "--height", str(payload["height"]),
        "--width", str(payload["width"]),
        "--num-frames", str(payload["num_frames"]),
        "--frame-rate", str(payload["frame_rate"]),
    ])

    if payload["quantization"]:
        command.extend(["--quantization", payload["quantization"]])
    if payload["enhance_prompt"]:
        command.append("--enhance-prompt")

    # Image conditioning: --image PATH FRAME_IDX STRENGTH [CRF]
    for img in resolved_images:
        args = [img["path"], str(img["frame_index"]), str(img["strength"])]
        if "crf" in img:
            args.append(str(img["crf"]))
        command.extend(["--image", *args])

    return command


def run_generation(payload: dict[str, Any]) -> dict[str, Any]:
    assets = ensure_assets()
    run_id = f"ltx-{int(time.time())}"
    work_dir = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=str(LTX_WORK_ROOT)))
    output_path = work_dir / "output.mp4"

    # Resolve image inputs (download URLs / decode base64)
    resolved_images = []
    for i, img_spec in enumerate(payload.get("raw_images") or []):
        resolved = resolve_image_input(img_spec, work_dir, i)
        resolved_images.append(resolved)
        LOGGER.info("resolved image %d: frame=%d strength=%.2f path=%s",
                     i, resolved["frame_index"], resolved["strength"], resolved["path"])

    command = build_command(assets, payload, output_path, resolved_images)

    env = os.environ.copy()
    env["HF_TOKEN"] = require_hf_token()
    env["PYTHONPATH"] = env.get("PYTHONPATH", str(LTX_REPO_DIR / "packages" / "ltx-core" / "src"))

    started_at = time.time()
    LOGGER.info("starting LTX job: %s (images: %d)", run_id, len(resolved_images))
    result = subprocess.run(
        command,
        cwd=str(LTX_REPO_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    runtime_sec = time.time() - started_at

    if result.returncode != 0:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(
            "LTX generation failed\n"
            f"stdout:\n{result.stdout[-8000:]}\n"
            f"stderr:\n{result.stderr[-8000:]}"
        )

    if not output_path.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError("LTX command completed but did not create output.mp4")

    persisted_path = LOCAL_OUTPUT_DIR / f"{run_id}.mp4"
    shutil.copy2(output_path, persisted_path)
    upload_url = maybe_upload_file(persisted_path, payload["output_upload_url"])
    inline_output = maybe_inline_output(persisted_path, payload["return_base64"])

    # Summarize image conditioning for response
    image_summary = [
        {"frame_index": img["frame_index"], "strength": img["strength"]}
        for img in resolved_images
    ]

    response = {
        "ok": True,
        "runtime_sec": round(runtime_sec, 3),
        "output_path": str(persisted_path),
        "output_upload_url": upload_url,
        "output_base64": inline_output,
        "output_size_bytes": persisted_path.stat().st_size,
        "output_sha256": sha256_file(persisted_path),
        "model": {
            "repo": LTX_MODEL_REPO,
            "checkpoint": LTX_CHECKPOINT_NAME,
            "spatial_upsampler": LTX_SPATIAL_UPSAMPLER_NAME,
            "gemma_repo": LTX_GEMMA_REPO,
            "repo_commit": LTX_REPO_COMMIT,
            "pipeline": payload.get("pipeline", "distilled"),
        },
        "input": {
            "prompt": payload["prompt"],
            "width": payload["width"],
            "height": payload["height"],
            "num_frames": payload["num_frames"],
            "frame_rate": payload["frame_rate"],
            "seed": payload["seed"],
            "quantization": payload["quantization"],
            "enhance_prompt": payload["enhance_prompt"],
            "duration_sec": round(payload["num_frames"] / payload["frame_rate"], 3),
            "images": image_summary,
        },
        "debug": {
            "command": command,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        },
    }
    shutil.rmtree(work_dir, ignore_errors=True)
    return response


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = validate_request(job.get("input") or {})
        return run_generation(payload)
    except InputError as exc:
        return {"ok": False, "error_type": "input_error", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("job failed")
        return {"ok": False, "error_type": "runtime_error", "error": str(exc)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
