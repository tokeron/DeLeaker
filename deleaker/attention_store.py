"""Rolling weighted average of attention maps across blocks/steps.

The deleaker uses this to smooth the entity-mask signal across time —
without it, per-block masks are too noisy to threshold cleanly.
"""

import torch


class AttentionStore:
    def __init__(self):
        self.aggregated_attention = None
        self.aggregation_counter = 0

    def update_aggregated_attention(
        self,
        attn_weight_after_softmax: torch.Tensor,
        start_aggregating_from: int,
        stop_aggregating_at: int,
        total_number_of_transformer_blocks: int,
        use_binary_history: bool = False,
        smooth_binary_mask: torch.Tensor = None,
    ) -> None:
        if total_number_of_transformer_blocks > stop_aggregating_at:
            return

        mean = torch.nanmean(attn_weight_after_softmax)
        attn_weight_after_softmax = torch.nan_to_num(attn_weight_after_softmax, nan=mean)

        if total_number_of_transformer_blocks == start_aggregating_from:
            self.aggregated_attention = attn_weight_after_softmax
        elif start_aggregating_from < total_number_of_transformer_blocks < stop_aggregating_at:
            n = self.aggregation_counter
            self.aggregated_attention = (
                self.aggregated_attention * (n / (n + 1))
                + (1 / (n + 1)) * attn_weight_after_softmax
            )

        self.aggregation_counter += 1

    def get_aggregated_attention(self) -> torch.Tensor:
        return self.aggregated_attention
