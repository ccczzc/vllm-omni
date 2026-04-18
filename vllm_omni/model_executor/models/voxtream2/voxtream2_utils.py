"""Shared helpers for Voxtream2 prompt and audio handling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

VOXTREAM2_DEFAULT_SAMPLE_RATE = 24_000


def estimate_voxtream2_prompt_len(text: str) -> int:
    return max(64, len(text) // 2 + 64)


def build_voxtream2_prompt(text: str, ref_audio_path: str) -> dict[str, Any]:
    return {
        "prompt_token_ids": [1] * estimate_voxtream2_prompt_len(text),
        "additional_information": {
            "text": [text],
            "ref_audio_path": ref_audio_path,
        },
    }


def flatten_voxtream2_audio_output(
    multimodal_output: Mapping[str, Any],
    *,
    default_sample_rate: int = VOXTREAM2_DEFAULT_SAMPLE_RATE,
    require_audio: bool = True,
) -> tuple[torch.Tensor, int]:
    audio = multimodal_output.get("audio")
    if audio is None:
        audio = multimodal_output.get("model_outputs")
    if audio is None:
        if require_audio:
            raise ValueError(f"No audio found in multimodal_output keys={list(multimodal_output.keys())}")
        return torch.zeros((0,), dtype=torch.float32), default_sample_rate

    if isinstance(audio, list):
        tensors = [torch.as_tensor(item).reshape(-1).float().cpu() for item in audio if item is not None]
        audio_tensor = torch.cat(tensors, dim=-1) if tensors else torch.zeros((0,), dtype=torch.float32)
    else:
        audio_tensor = torch.as_tensor(audio).reshape(-1).float().cpu()

    sr_raw = multimodal_output.get("sr", default_sample_rate)
    if isinstance(sr_raw, list):
        sr_raw = sr_raw[-1] if sr_raw else default_sample_rate
    sample_rate = int(sr_raw.item()) if hasattr(sr_raw, "item") else int(sr_raw)
    return audio_tensor, sample_rate
