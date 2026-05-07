"""Vanilla FLUX.1-dev inference (no deleaker) — for baseline comparison.

Run from the repo root:
    python examples/vanilla_inference.py
"""

import os
from pathlib import Path

import torch
from diffusers import FluxPipeline


PROMPT = "A bat is sitting on a branch while an owl is flying above through the moonlit forest."
SEED = 200
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 3.5
HEIGHT = WIDTH = 1024


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(SEED)
    image = pipe(
        prompt=PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
        max_sequence_length=256,
    ).images[0]

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vanilla_seed{SEED}.png"
    image.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
