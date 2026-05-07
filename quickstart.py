"""Minimal one-prompt deleaker example.

Run from the repo root after `pip install -e .`:

    python quickstart.py

You'll need access to FLUX.1-dev on Hugging Face (it is gated):
sign in at https://huggingface.co/black-forest-labs/FLUX.1-dev,
accept the license, and run `huggingface-cli login` once.
"""

import torch

from deleaker import DeleakerFluxPipeline


PROMPT = "A bat and an owl are perched side by side on a tree branch."
ENTITIES = ["bat", "owl"]
SEED = 200


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DeleakerFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
    ).to(device)

    image = pipe(
        prompt=PROMPT,
        entities=ENTITIES,
        num_inference_steps=20,
        guidance_scale=3.5,
        height=512,
        width=512,
        generator=torch.Generator(device=device).manual_seed(SEED),
        max_sequence_length=256,
    ).images[0]

    out_path = "out.png"
    image.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
