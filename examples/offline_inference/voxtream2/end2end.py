"""Offline inference demo for Voxtream2 TTS via vLLM Omni."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import soundfile as sf
import torch
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.voxtream2.voxtream2_import_utils import ensure_voxtream_available

ensure_voxtream_available()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "herimor/voxtream2"
DEFAULT_STAGE_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "vllm_omni",
    "model_executor",
    "stage_configs",
    "voxtream2_1stage.yaml",
)


def _estimate_prompt_len(text: str) -> int:
    return max(64, len(text) // 2 + 64)


def _build_prompt(text: str, ref_audio: str) -> dict:
    return {
        "prompt_token_ids": [1] * _estimate_prompt_len(text),
        "additional_information": {
            "text": [text],
            "ref_audio_path": ref_audio,
        },
    }


def _flatten_audio(multimodal_output: dict) -> tuple[torch.Tensor, int]:
    audio = multimodal_output.get("audio")
    if audio is None:
        audio = multimodal_output.get("model_outputs")
    if audio is None:
        raise ValueError(f"No audio found in multimodal_output keys={list(multimodal_output.keys())}")

    if isinstance(audio, list):
        if not audio:
            audio_tensor = torch.zeros((0,), dtype=torch.float32)
        else:
            audio_tensor = torch.cat([a.reshape(-1).float() for a in audio], dim=-1)
    elif isinstance(audio, torch.Tensor):
        audio_tensor = audio.reshape(-1).float()
    else:
        audio_tensor = torch.tensor(audio, dtype=torch.float32).reshape(-1)

    sr_raw = multimodal_output.get("sr", 24000)
    if isinstance(sr_raw, list):
        sr_raw = sr_raw[-1] if sr_raw else 24000
    sample_rate = int(sr_raw.item()) if hasattr(sr_raw, "item") else int(sr_raw)
    return audio_tensor, sample_rate


def parse_args():
    parser = FlexibleArgumentParser(description="Voxtream2 offline inference")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path or HF repo.")
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=DEFAULT_STAGE_CONFIG,
        help="Path to stage config yaml.",
    )
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize.")
    parser.add_argument("--ref-audio", type=str, required=True, help="Reference audio path.")
    parser.add_argument("--output", type=str, required=True, help="Output wav path.")
    parser.add_argument(
        "--voxtream-root",
        type=str,
        default=None,
        help="Optional local voxtream repo path that contains configs/*.json.",
    )
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument("--stage-init-timeout", type=int, default=300)
    return parser.parse_args()


def main(args):
    if not Path(args.ref_audio).is_file():
        raise FileNotFoundError(f"Reference audio not found: {args.ref_audio}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.voxtream_root:
        os.environ["VOXTREAM2_ROOT"] = args.voxtream_root
        os.environ.setdefault("VOXTREAM_ROOT", args.voxtream_root)
        logger.info("Set VOXTREAM2_ROOT=%s for split config lookup", args.voxtream_root)

    prompt = _build_prompt(args.text, args.ref_audio)
    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    try:
        for stage_outputs in omni.generate([prompt]):
            request_output = stage_outputs.request_output
            if request_output is None or not request_output.outputs:
                continue
            mm = request_output.outputs[0].multimodal_output
            audio_tensor, sample_rate = _flatten_audio(mm)
            sf.write(str(output_path), audio_tensor.cpu().numpy(), sample_rate, format="WAV")
            logger.info("Saved %d samples @ %d Hz to %s", audio_tensor.numel(), sample_rate, output_path)
            print(f"Saved audio to {output_path}")
            return
    finally:
        omni.close()

    raise RuntimeError("No audio output returned by Voxtream2 pipeline.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_args())
