"""FLUX.1-dev inference *with* the deleaker intervention.

Run from the repo root:
    python examples/deleaker_inference.py
"""

from pathlib import Path

import torch

from deleaker import DeleakerFluxPipeline, DeleakerConfig


PROMPT = "A cat and a cheetah are splashing each other with water in a shallow river."
ENTITIES = ["cat", "cheetah"]
SEED = 300
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 3.5
HEIGHT = WIDTH = 1024


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DeleakerFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
    ).to(device)

    # Tune any of these knobs by passing a custom DeleakerConfig instance.
    cfg = DeleakerConfig(
        std_mul_text=1.0,
        std_mul_image_image=1.0,
        k_strength=1.2,
        start_intervention_from=18,
        stop_intervention_at=NUM_INFERENCE_STEPS * 57,
        stop_aggregating_at=NUM_INFERENCE_STEPS * 57,
    )

    generator = torch.Generator(device=device).manual_seed(SEED)
    image = pipe(
        prompt=PROMPT,
        entities=ENTITIES,
        deleaker_config=cfg,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
        max_sequence_length=256,
    ).images[0]

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"deleaker_seed{SEED}.png"
    image.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
