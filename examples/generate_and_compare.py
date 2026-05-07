"""Generate clean (vanilla) + deleaker outputs side-by-side, with grids.

Two modes:

* Default (HF mode) — reads prompts from the published `tokeron/slim-dataset`,
  uses every seed listed in the dataset for each selected prompt, and
  builds a 3-panel grid per (prompt, seed): ``reference | vanilla | deleaker``.
  The reference is the leaky image already on the Hub; the locally
  regenerated vanilla should match it (sanity check).

* Custom mode (``--prompt`` + ``--entities``) — runs an arbitrary user prompt
  with comma-separated entities at one or more seeds (default 100,200,300).
  The grid is 2-panel ``vanilla | deleaker`` since there's no reference.

Examples:

    # HF mode, default prompt indices [2, 3, 4]
    python examples/generate_and_compare.py

    # HF mode, choose your own prompt indices
    python examples/generate_and_compare.py --hf-indices 0,1,2

    # Custom prompt
    python examples/generate_and_compare.py \\
        --prompt "A horse is doing ski, while a zebra is falling from the mountain." \\
        --entities "horse,zebra" \\
        --seeds 100,200,300
"""

import argparse
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from deleaker import DeleakerFluxPipeline, DeleakerConfig


HF_DATASET = "tokeron/slim-dataset"
DEFAULT_HF_INDICES = [2, 3, 4]
DEFAULT_CUSTOM_SEEDS = [100, 200, 300]
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 3.5
# The dataset's reference images were generated at 512x512 — same-seed runs
# at any other resolution would produce wholly different compositions.
HEIGHT = WIDTH = 512


def slug(prompt: str, max_words: int = 8) -> str:
    cleaned = re.sub(r"[^\w\s]", "", prompt.lower()).strip()
    return "_".join(cleaned.split()[:max_words])


def gather_prompts_by_index(ds, indices: list) -> list:
    """Return planning records for prompts at the given 0-based indices.

    Indices are positions in the order in which unique prompts first appear
    in the dataset. Each record has ``prompt_idx``, ``prompt``, ``entities``,
    ``seeds``, ``references``.
    """
    wanted = set(indices)
    seen: dict = {}
    by_prompt: dict = {}
    for row in ds:
        p = row["prompt"]
        if p not in seen:
            seen[p] = len(seen)
            if seen[p] in wanted:
                by_prompt[p] = {
                    "prompt_idx": seen[p],
                    "prompt": p,
                    "entities": [e.strip() for e in row["entities"].split(",") if e.strip()],
                    "seeds": [],
                    "references": [],
                }
        if seen[p] in wanted:
            by_prompt[p]["seeds"].append(int(row["seed"]))
            by_prompt[p]["references"].append(row["image"])
    return sorted(by_prompt.values(), key=lambda r: r["prompt_idx"])


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label(img: Image.Image, text: str, height: int = 32) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img, (0, height))
    draw = ImageDraw.Draw(out)
    font = _load_font(int(height * 0.6))
    draw.text((6, 4), text, fill="black", font=font)
    return out


def build_pair_grid(vanilla: Image.Image, deleaker: Image.Image, seed: int,
                    reference: Image.Image = None) -> Image.Image:
    panels = []
    if reference is not None:
        panels.append(label(reference, f"reference (HF) | seed {seed}"))
    panels.append(label(vanilla, f"vanilla | seed {seed}"))
    panels.append(label(deleaker, f"deleaker | seed {seed}"))
    w = sum(p.width for p in panels)
    h = max(p.height for p in panels)
    grid = Image.new("RGB", (w, h), "white")
    x = 0
    for p in panels:
        grid.paste(p, (x, 0))
        x += p.width
    return grid


def build_prompt_grid(pair_grids: list, prompt: str, entities: list) -> Image.Image:
    if not pair_grids:
        raise ValueError("empty pair_grids")
    w = pair_grids[0].width
    header_h = 48
    header = Image.new("RGB", (w, header_h), "white")
    draw = ImageDraw.Draw(header)
    font = _load_font(20)
    draw.text((10, 6), f"prompt: {prompt}", fill="black", font=font)
    draw.text((10, 26), f"entities: {', '.join(entities)}", fill="black", font=font)
    h = header_h + sum(g.height for g in pair_grids)
    grid = Image.new("RGB", (w, h), "white")
    grid.paste(header, (0, 0))
    y = header_h
    for g in pair_grids:
        grid.paste(g, (0, y))
        y += g.height
    return grid


def build_plan_from_hf(indices: list) -> list:
    print(f"Loading {HF_DATASET}...")
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, split="train")
    return gather_prompts_by_index(ds, indices)


def build_plan_custom(prompt: str, entities: list, seeds: list) -> list:
    return [{
        "prompt_idx": 0,
        "prompt": prompt,
        "entities": entities,
        "seeds": list(seeds),
        "references": [None] * len(seeds),
    }]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", default=None,
                   help="Custom prompt. If set, --entities is required and --hf-indices is ignored.")
    p.add_argument("--entities", default=None,
                   help='Comma-separated entities present verbatim in the prompt, e.g. "horse,zebra".')
    p.add_argument("--seeds", default=None,
                   help=f"Comma-separated seeds for custom mode (default {DEFAULT_CUSTOM_SEEDS}).")
    p.add_argument("--hf-indices", default=None,
                   help=f"Comma-separated 0-based indices of unique prompts in the HF dataset "
                        f"(default {DEFAULT_HF_INDICES}). Ignored if --prompt is given.")
    p.add_argument("--out-subdir", default=None,
                   help="Subdir of examples/output/ to write to. Defaults to "
                        "'hf_dataset' in HF mode, 'custom' in custom mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompt is not None:
        if not args.entities:
            raise SystemExit("--entities is required when --prompt is given.")
        entities = [e.strip() for e in args.entities.split(",") if e.strip()]
        seeds = (
            [int(s) for s in args.seeds.split(",")] if args.seeds else DEFAULT_CUSTOM_SEEDS
        )
        plan = build_plan_custom(args.prompt, entities, seeds)
        out_subdir = args.out_subdir or "custom"
        mode = "custom"
    else:
        indices = (
            [int(s) for s in args.hf_indices.split(",")] if args.hf_indices else DEFAULT_HF_INDICES
        )
        plan = build_plan_from_hf(indices)
        out_subdir = args.out_subdir or "hf_dataset"
        mode = "hf"

    total = sum(len(p["seeds"]) for p in plan)
    print(f"Mode: {mode}. {len(plan)} prompts, {total} (prompt, seed) pairs "
          f"-> generating {2 * total} images (vanilla + deleaker).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DeleakerFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16,
    ).to(device)

    out_root = Path(__file__).parent / "output" / out_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = DeleakerConfig()

    counter = 0
    for k, item in enumerate(plan):
        prompt = item["prompt"]
        entities = item["entities"]
        prompt_dir = out_root / f"{item['prompt_idx']:04d}_{slug(prompt)}"
        prompt_dir.mkdir(exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(
            f"prompt: {prompt}\nentities: {', '.join(entities)}\n"
        )

        print(f"\n[{k + 1}/{len(plan)}] (idx {item['prompt_idx']}) {prompt!r}")
        print(f"            entities={entities}  seeds={item['seeds']}")

        pair_grids = []
        for seed, ref_image in zip(item["seeds"], item["references"]):
            counter += 1
            print(f"  ({counter}/{total}) seed={seed}: vanilla + deleaker...")

            # vanilla — the deleaker pipeline with use_deleaker=False delegates
            # straight to the stock FluxPipeline path, so this is a baseline at
            # exactly the same params as the deleaker run below.
            g = torch.Generator(device=device).manual_seed(seed)
            img_vanilla = pipe(
                prompt=prompt,
                use_deleaker=False,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=HEIGHT, width=WIDTH,
                generator=g,
                max_sequence_length=256,
            ).images[0]

            g = torch.Generator(device=device).manual_seed(seed)
            img_deleaker = pipe(
                prompt=prompt,
                entities=entities,
                deleaker_config=cfg,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=HEIGHT, width=WIDTH,
                generator=g,
                max_sequence_length=256,
            ).images[0]

            if ref_image is not None:
                ref_image.save(prompt_dir / f"seed_{seed:04d}_reference.png")
            img_vanilla.save(prompt_dir / f"seed_{seed:04d}_vanilla.png")
            img_deleaker.save(prompt_dir / f"seed_{seed:04d}_deleaker.png")

            pair = build_pair_grid(img_vanilla, img_deleaker, seed, reference=ref_image)
            pair.save(prompt_dir / f"seed_{seed:04d}_compare.png")
            pair_grids.append(pair)

        master = build_prompt_grid(pair_grids, prompt, entities)
        master.save(prompt_dir / "grid.png")
        print(f"  -> {prompt_dir / 'grid.png'}")

    print(f"\nDone. Wrote {2 * total} generated images and {total} pair "
          f"grids + {len(plan)} prompt grids under {out_root}")


if __name__ == "__main__":
    main()
