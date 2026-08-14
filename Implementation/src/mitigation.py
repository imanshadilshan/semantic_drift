"""Drift mitigation strategies: masked conditioning, region-locking, and (stretch) attention-restricted editing."""


def masked_conditioning(image, mask, instruction: str):
    raise NotImplementedError


def region_locking(pre_image, post_image, mask):
    raise NotImplementedError


def attention_restricted_editing(image, mask, instruction: str):
    raise NotImplementedError
