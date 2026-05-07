from dataclasses import dataclass, asdict


@dataclass
class DeleakerConfig:
    """Knobs for the deleaker attention intervention.

    All defaults are tuned for FLUX.1-dev with `num_inference_steps=20`
    and reproduce the configuration used in the paper figures.

    The intervention window is measured in absolute *block-step* index,
    i.e. ``step_index * num_blocks_per_step + block_index_within_step``.
    For FLUX-dev each denoising step runs 19 double + 38 single = 57 blocks,
    so ``stop_intervention_at = num_inference_steps * 57`` covers all blocks.
    """

    # Master switch.
    use_deleaker: bool = True

    # Threshold multipliers for entity masks (mean + k * std).
    std_mul_text: float = 1.0
    std_mul_image_image: float = 1.0

    # How much to amplify self-entity image-text attention.
    k_strength: float = 1.2

    # Window for building the rolling entity-mask history.
    start_aggregating_from: int = 12
    stop_aggregating_at: int = 12 * 57

    # Window for actually editing attention.
    start_intervention_from: int = 18
    stop_intervention_at: int = 20 * 57

    # History smoothing.
    use_history: bool = True
    use_binary_history: bool = False
    history_sliding_window: int = 10
    do_smoothing: bool = True

    # Ablation switches.
    do_image_image: bool = True
    do_image_text_strengthening: bool = True
    do_image_text_weakening: bool = True
    do_text_text_weakening: bool = False

    # Number of text tokens (FLUX uses 256/512 by max_sequence_length).
    num_text_tokens: int = 256

    def to_dict(self) -> dict:
        return asdict(self)
