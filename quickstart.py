"""Minimal one-prompt deleaker example.

Run from the repo root after `pip install -e .`:

    # default bat/owl prompt
    python quickstart.py

    # your own prompt + entities (entities must appear verbatim in prompt)
    python quickstart.py --prompt "A cow and a horse are sitting together in a bus." --entities "cow,horse"

    # vanilla baseline (no deleaker) for comparison
    python quickstart.py --no-use-deleaker --out vanilla.png

You'll need access to FLUX.1-dev on Hugging Face (it is gated):
sign in at https://huggingface.co/black-forest-labs/FLUX.1-dev,
accept the license, and run `huggingface-cli login` once.
"""

import argparse

import torch

from deleaker import DeleakerFluxPipeline


DEFAULT_PROMPT = "A bat and an owl are perched side by side on a tree branch."
DEFAULT_ENTITIES = "bat,owl"
DEFAULT_SEED = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Text prompt. Each item in --entities must appear verbatim in it.")
    p.add_argument("--entities", default=DEFAULT_ENTITIES,
                   help='Comma-separated entities to deleak, e.g. "cat,cheetah".')
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--use-deleaker", action=argparse.BooleanOptionalAction, default=True,
                   help="Use --no-deleaker to run vanilla FLUX.1-dev for comparison.")
    p.add_argument("--out", default="out.png", help="Output PNG path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    entities = [e.strip() for e in args.entities.split(",") if e.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DeleakerFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
    ).to(device)

    image = pipe(
        prompt=args.prompt,
        entities=entities,
        use_deleaker=args.use_deleaker,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        max_sequence_length=256,
    ).images[0]

    image.save(args.out)
    print(f"Saved {args.out}  (prompt={args.prompt!r}  entities={entities}  seed={args.seed}  deleaker={args.use_deleaker})")


if __name__ == "__main__":
    main()
