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
DEFAULT_HF_INDICES = [1, 3, 4]
DEFAULT_CUSTOM_SEEDS = [100, 200, 300]
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 3.5
# The HF dataset's reference images were generated at 512x512. In HF mode the
# resolution must stay at 512 for the reference panel to match the locally
# regenerated vanilla; in custom mode any resolution works.
DEFAULT_HEIGHT = DEFAULT_WIDTH = 512

# ---- DeleakerConfig defaults (edit here, or override on the CLI) ----
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


def build_pair_grid(vanilla: Image.Image, deleaker: Image.Image, seed: int) -> Image.Image:
    a = label(vanilla, f"vanilla | seed {seed}")
    b = label(deleaker, f"deleaker | seed {seed}")
    grid = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), "white")
    grid.paste(a, (0, 0))
    grid.paste(b, (a.width, 0))
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
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                   help=f"Image height (default {DEFAULT_HEIGHT}). HF mode "
                        f"requires 512 to match the reference images.")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                   help=f"Image width (default {DEFAULT_WIDTH}). HF mode "
                        f"requires 512 to match the reference images.")

    # DeleakerConfig knobs (defaults pulled from the module-level constants
    # so they show up in --help and can be tweaked in one place).
    g = p.add_argument_group("deleaker config")
    g.add_argument("--use-deleaker", action=argparse.BooleanOptionalAction,
                   default=USE_DELEAKER, help="Master on/off for the intervention.")
    g.add_argument("--std-mul-text", type=float, default=STD_MUL_TEXT)
    g.add_argument("--std-mul-image-image", type=float, default=STD_MUL_IMAGE_IMAGE)
    g.add_argument("--k-strength", type=float, default=K_STRENGTH)
    g.add_argument("--start-aggregating-from", type=int, default=START_AGGREGATING_FROM)
    g.add_argument("--stop-aggregating-at", type=int, default=STOP_AGGREGATING_AT)
    g.add_argument("--start-intervention-from", type=int, default=START_INTERVENTION_FROM)
    g.add_argument("--stop-intervention-at", type=int, default=STOP_INTERVENTION_AT)
    g.add_argument("--use-history", action=argparse.BooleanOptionalAction,
                   default=USE_HISTORY)
    g.add_argument("--use-binary-history", action=argparse.BooleanOptionalAction,
                   default=USE_BINARY_HISTORY)
    g.add_argument("--history-sliding-window", type=int, default=HISTORY_SLIDING_WINDOW)
    g.add_argument("--do-smoothing", action=argparse.BooleanOptionalAction,
                   default=DO_SMOOTHING)
    g.add_argument("--do-image-image", action=argparse.BooleanOptionalAction,
                   default=DO_IMAGE_IMAGE)
    g.add_argument("--do-image-text-strengthening", action=argparse.BooleanOptionalAction,
                   default=DO_IMAGE_TEXT_STRENGTHENING)
    g.add_argument("--do-image-text-weakening", action=argparse.BooleanOptionalAction,
                   default=DO_IMAGE_TEXT_WEAKENING)
    g.add_argument("--do-text-text-weakening", action=argparse.BooleanOptionalAction,
                   default=DO_TEXT_TEXT_WEAKENING)
    g.add_argument("--num-text-tokens", type=int, default=NUM_TEXT_TOKENS)
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

    cfg = DeleakerConfig(
        use_deleaker=args.use_deleaker,
        std_mul_text=args.std_mul_text,
        std_mul_image_image=args.std_mul_image_image,
        k_strength=args.k_strength,
        start_aggregating_from=args.start_aggregating_from,
        stop_aggregating_at=args.stop_aggregating_at,
        start_intervention_from=args.start_intervention_from,
        stop_intervention_at=args.stop_intervention_at,
        use_history=args.use_history,
        use_binary_history=args.use_binary_history,
        history_sliding_window=args.history_sliding_window,
        do_smoothing=args.do_smoothing,
        do_image_image=args.do_image_image,
        do_image_text_strengthening=args.do_image_text_strengthening,
        do_image_text_weakening=args.do_image_text_weakening,
        do_text_text_weakening=args.do_text_text_weakening,
        num_text_tokens=args.num_text_tokens,
    )

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
                height=args.height, width=args.width,
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
                height=args.height, width=args.width,
                generator=g,
                max_sequence_length=256,
            ).images[0]

            if ref_image is not None:
                ref_image.save(prompt_dir / f"seed_{seed:04d}_reference.png")
            img_vanilla.save(prompt_dir / f"seed_{seed:04d}_vanilla.png")
            img_deleaker.save(prompt_dir / f"seed_{seed:04d}_deleaker.png")

            pair = build_pair_grid(img_vanilla, img_deleaker, seed)
            pair.save(prompt_dir / f"seed_{seed:04d}_compare.png")
            pair_grids.append(pair)

        master = build_prompt_grid(pair_grids, prompt, entities)
        master.save(prompt_dir / "grid.png")
        print(f"  -> {prompt_dir / 'grid.png'}")

    print(f"\nDone. Wrote {2 * total} generated images and {total} pair "
          f"grids + {len(plan)} prompt grids under {out_root}")


if __name__ == "__main__":
    main()
