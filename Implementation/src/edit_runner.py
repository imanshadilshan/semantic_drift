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


def run_edit_chain(image: Image.Image, instructions: list[str], mitigation=None) -> list[Image.Image]:
    """Applies instructions sequentially, each step's output feeding the next.

    Returns one image per step (does not include the original). If mitigation is given, it's
    called as mitigation(pre_step_image, raw_edited_image) -> corrected_image after every step.
    """
    outputs = []
    current = image
    for instruction in instructions:
        edited = run_edit(current, instruction)
        if mitigation is not None:
            edited = mitigation(current, edited)
        outputs.append(edited)
        current = edited
    return outputs
