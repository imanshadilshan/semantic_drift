"""Thin wrapper around the InstructPix2Pix diffusers pipeline. GPU-only, heavy."""

from PIL import Image

from .utils import get_device

_pipe = None


def _get_pipeline(checkpoint: str = "timbrooks/instruct-pix2pix"):
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

        device = get_device()
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        _pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(checkpoint, torch_dtype=dtype)
        _pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(_pipe.scheduler.config)
        _pipe.to(device)
    return _pipe


def run_edit(
    image: Image.Image,
    instruction: str,
    checkpoint: str = "timbrooks/instruct-pix2pix",
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.5,
) -> Image.Image:
    pipe = _get_pipeline(checkpoint)
    return pipe(
        instruction,
        image=image,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
    ).images[0]


def run_edit_chain(image: Image.Image, instructions: list[str], step_fn=None) -> list[Image.Image]:
    """Applies instructions sequentially, each step's output feeding the next.

    Returns one image per step (does not include the original). step_fn(pre_image, instruction) ->
    edited_image defaults to plain run_edit (no mitigation). Mitigation strategies in mitigation.py
    are full step functions rather than post-hoc correctors, since masked conditioning needs to
    control what the model is given *before* generation, not just clean up its output after.
    """
    step_fn = step_fn or run_edit
    outputs = []
    current = image
    for instruction in instructions:
        current = step_fn(current, instruction)
        outputs.append(current)
    return outputs
