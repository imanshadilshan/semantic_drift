"""Thin wrapper around the InstructPix2Pix diffusers pipeline. GPU-only, heavy."""


def run_edit(image, instruction: str, checkpoint: str = "timbrooks/instruct-pix2pix"):
    raise NotImplementedError


def run_edit_chain(image, instructions: list[str], mitigation=None):
    raise NotImplementedError
