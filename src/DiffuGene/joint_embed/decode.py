#!/usr/bin/env python
"""
Alias for joint_embed inference/decoding functionality.
"""

from .infer import inference

# Re-export the main inference function
decode_latents = inference

__all__ = ["decode_latents", "inference"]
