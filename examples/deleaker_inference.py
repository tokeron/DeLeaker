"""FLUX.1-dev inference *with* the deleaker intervention.

Run from the repo root:
    python examples/deleaker_inference.py
"""

from pathlib import Path

import torch

from deleaker import DeleakerFluxPipeline, DeleakerConfig


PROMPT = "A bat is sitting on a branch while an owl is flying above through the moonlit forest."
ENTITIES = ["bat", "owl"]
SEED = 200
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 3.5
HEIGHT = WIDTH = 512

# ---- DeleakerConfig knobs (edit here to tweak) ----
USE_DELEAKER = True
STD_MUL_TEXT = 1.0                 # threshold for image-text entity mask: mean + k*std
STD_MUL_IMAGE_IMAGE = 1.0          # threshold for image-image cross-entity mask
K_STRENGTH = 1.2                   # self-entity image-text boost
START_AGGREGATING_FROM = 12        # block-step where rolling history begins
STOP_AGGREGATING_AT = 12 * 57      # block-step where history stops
START_INTERVENTION_FROM = 18       # block-step where attention editing begins
STOP_INTERVENTION_AT = 20 * 57     # block-step where editing stops
USE_HISTORY = True
USE_BINARY_HISTORY = False
HISTORY_SLIDING_WINDOW = 10
DO_SMOOTHING = True                # morphological clean of entity masks
# Ablation switches:
DO_IMAGE_IMAGE = True
DO_IMAGE_TEXT_STRENGTHENING = True
DO_IMAGE_TEXT_WEAKENING = True
DO_TEXT_TEXT_WEAKENING = False
NUM_TEXT_TOKENS = 256


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DeleakerFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
    ).to(device)

    cfg = DeleakerConfig(
        use_deleaker=USE_DELEAKER,
        std_mul_text=STD_MUL_TEXT,
        std_mul_image_image=STD_MUL_IMAGE_IMAGE,
        k_strength=K_STRENGTH,
        start_aggregating_from=START_AGGREGATING_FROM,
        stop_aggregating_at=STOP_AGGREGATING_AT,
        start_intervention_from=START_INTERVENTION_FROM,
        stop_intervention_at=STOP_INTERVENTION_AT,
        use_history=USE_HISTORY,
        use_binary_history=USE_BINARY_HISTORY,
        history_sliding_window=HISTORY_SLIDING_WINDOW,
        do_smoothing=DO_SMOOTHING,
        do_image_image=DO_IMAGE_IMAGE,
        do_image_text_strengthening=DO_IMAGE_TEXT_STRENGTHENING,
        do_image_text_weakening=DO_IMAGE_TEXT_WEAKENING,
        do_text_text_weakening=DO_TEXT_TEXT_WEAKENING,
        num_text_tokens=NUM_TEXT_TOKENS,
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
