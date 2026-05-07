"""Deleaker FLUX attention processor.

Drop-in replacement for diffusers' ``FluxAttnProcessor2_0`` that, on each
attention call, optionally rewrites the attention logits to suppress
cross-entity leakage. The math is identical to the reference paper
implementation; visualization/statistics code paths have been removed
for clarity.
"""

from itertools import combinations
from typing import Optional
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


# Stock FLUX has 19 double + 38 single = 57 transformer blocks per denoising step.
_NUM_DOUBLE_BLOCKS = 19
_NUM_BLOCKS_PER_STEP = 57


class DeleakerFluxAttnProcessor2_0:
    """Replaces ``FluxAttnProcessor2_0`` and rewrites attention to reduce leakage.

    Pass an instance of this processor (one per attention layer) to
    ``transformer.set_attn_processor(...)``. Each processor parses its own
    ``layer_name`` (e.g. ``"transformer_blocks.3"``) to derive its absolute
    block offset within a denoising step.

    Per-call inputs come in via ``deleaker_kwargs`` (typically threaded from
    the pipeline through ``joint_attention_kwargs``):

    - ``token_entity_indices``: ``{entity: range(start, end)}`` token indices
    - ``step_index``: current denoising step (0-indexed)
    - all knobs from :class:`deleaker.config.DeleakerConfig`
    """

    def __init__(self, layer_name: str = "", attention_store=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("DeleakerFluxAttnProcessor2_0 requires PyTorch >= 2.0")
        self.layer_name = layer_name
        self.attention_store = attention_store
        if layer_name.startswith("single_transformer_blocks"):
            self.transformer_type = "single"
            block_idx = int(layer_name.split(".")[1])
            self.block_offset = _NUM_DOUBLE_BLOCKS + block_idx
        else:
            self.transformer_type = "double"
            self.block_offset = int(layer_name.split(".")[1])

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        deleaker_kwargs: Optional[dict] = None,
    ) -> torch.FloatTensor:
        deleaker_kwargs = deleaker_kwargs or {}
        use_deleaker = deleaker_kwargs.get("use_deleaker", False)
        token_entity_indices = deleaker_kwargs.get("token_entity_indices", None)

        # If deleaker is off or there are no entity indices, fall back to the
        # stock SDPA path.
        if not use_deleaker or not token_entity_indices:
            return self._standard_forward(
                attn, hidden_states, encoder_hidden_states, image_rotary_emb
            )

        # Pull the rest of the knobs.
        step_index = deleaker_kwargs["step_index"]
        start_aggregating_from = deleaker_kwargs["start_aggregating_from"]
        stop_aggregating_at = deleaker_kwargs["stop_aggregating_at"]
        start_intervention_from = deleaker_kwargs["start_intervention_from"]
        stop_intervention_at = deleaker_kwargs["stop_intervention_at"]
        std_mul_text = deleaker_kwargs["std_mul_text"]
        std_mul_image_image = deleaker_kwargs["std_mul_image_image"]
        k_strength = deleaker_kwargs["k_strength"]
        use_history = deleaker_kwargs["use_history"]
        use_binary_history = deleaker_kwargs["use_binary_history"]
        do_smoothing = deleaker_kwargs["do_smoothing"]
        do_image_image = deleaker_kwargs["do_image_image"]
        do_image_text_strengthening = deleaker_kwargs["do_image_text_strengthening"]
        do_image_text_weakening = deleaker_kwargs["do_image_text_weakening"]
        do_text_text_weakening = deleaker_kwargs["do_text_text_weakening"]
        num_text_tokens = deleaker_kwargs["num_text_tokens"]

        total_number_of_transformer_blocks = (
            _NUM_BLOCKS_PER_STEP * step_index + self.block_offset
        )
        do_intervention = (
            start_intervention_from
            <= total_number_of_transformer_blocks
            <= stop_intervention_at
        )

        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # Project + reshape to (B, H, L, D).
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_q = attn.add_q_proj(encoder_hidden_states)
            encoder_k = attn.add_k_proj(encoder_hidden_states)
            encoder_v = attn.add_v_proj(encoder_hidden_states)
            encoder_q = encoder_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_k = encoder_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_v = encoder_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                encoder_q = attn.norm_added_q(encoder_q)
            if attn.norm_added_k is not None:
                encoder_k = attn.norm_added_k(encoder_k)
            query = torch.cat([encoder_q, query], dim=2)
            key = torch.cat([encoder_k, key], dim=2)
            value = torch.cat([encoder_v, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Materialize the attention logits explicitly (we need pre-softmax
        # logits to add -inf masks, then renormalize).
        attn_weight = self._scaled_dot_product_logits(query, key)
        attn_weight_after_softmax = attn_weight.softmax(dim=-1)
        if torch.isnan(attn_weight).all():
            raise ValueError(
                f"All values in attention are NaN at step {step_index}, "
                f"layer {self.layer_name}"
            )

        bs, num_heads, _, _ = attn_weight.shape

        if total_number_of_transformer_blocks >= start_aggregating_from or use_binary_history:
            # Update rolling history (if requested).
            if use_history and not use_binary_history and total_number_of_transformer_blocks >= start_aggregating_from:
                self.attention_store.update_aggregated_attention(
                    attn_weight_after_softmax,
                    start_aggregating_from=start_aggregating_from,
                    stop_aggregating_at=stop_aggregating_at,
                    total_number_of_transformer_blocks=total_number_of_transformer_blocks,
                )

            if do_intervention or use_binary_history:
                # ---- Build per-entity image masks from image-text attention ----
                image_latent_indices_per_entity = {
                    entity: {"prompt_indices": list(indices), "image_mask": None}
                    for entity, indices in token_entity_indices.items()
                }
                entity_masks: dict = {}

                for entity, props in image_latent_indices_per_entity.items():
                    prompt_indices = props["prompt_indices"]
                    if use_history and total_number_of_transformer_blocks > start_aggregating_from:
                        agg = self.attention_store.get_aggregated_attention()
                        image_query_entity_key = agg[:, :, num_text_tokens:, prompt_indices]
                    else:
                        image_query_entity_key = attn_weight_after_softmax[:, :, num_text_tokens:, prompt_indices]

                    # Replace NaNs with mean of finite values.
                    finite = image_query_entity_key[~torch.isnan(image_query_entity_key)]
                    fill = finite.mean() if finite.numel() > 0 else 0
                    image_query_entity_key = torch.where(
                        torch.isnan(image_query_entity_key), fill, image_query_entity_key
                    )

                    # Average across the entity's sub-tokens, then heads.
                    image_query_entity_key = image_query_entity_key.mean(dim=-1)
                    avg_over_heads = image_query_entity_key.mean(dim=1)

                    # Threshold: mean + std_mul_text * std.
                    mean = avg_over_heads.mean()
                    std = avg_over_heads.std()
                    threshold = mean + std_mul_text * std
                    entity_mask = avg_over_heads > threshold

                    if do_smoothing:
                        device = entity_mask.device
                        smoothed = []
                        for i in range(entity_mask.shape[0]):
                            smoothed.append(_morphological_clean(entity_mask[i]))
                        entity_mask = torch.tensor(np.stack(smoothed)).squeeze(-1).bool().to(device)

                    props["image_mask"] = entity_mask
                    entity_masks[entity] = entity_mask

                if use_binary_history and total_number_of_transformer_blocks >= start_aggregating_from:
                    first_entity = next(iter(token_entity_indices))
                    smooth_binary_mask = torch.zeros_like(entity_masks[first_entity])
                    for m in entity_masks.values():
                        smooth_binary_mask = smooth_binary_mask | m
                    self.attention_store.update_aggregated_attention(
                        attn_weight_after_softmax,
                        start_aggregating_from=start_aggregating_from,
                        stop_aggregating_at=stop_aggregating_at,
                        total_number_of_transformer_blocks=total_number_of_transformer_blocks,
                        use_binary_history=use_binary_history,
                        smooth_binary_mask=smooth_binary_mask,
                    )

                # ---- Apply the intervention to attn_weight ----
                if do_intervention:
                    for first_entity, second_entity in combinations(token_entity_indices.keys(), 2):
                        e1 = image_latent_indices_per_entity[first_entity]
                        e2 = image_latent_indices_per_entity[second_entity]
                        e1_image = e1["image_mask"]
                        e2_image = e2["image_mask"]
                        if e1_image.shape[-1] == 0 or e2_image.shape[-1] == 0:
                            raise ValueError(
                                f"Empty image mask for {first_entity!r} or {second_entity!r}"
                            )

                        # ---- image-text: weaken cross-entity, strengthen self-entity ----
                        if do_image_text_weakening:
                            attn_weight[:, :, num_text_tokens:, e2["prompt_indices"]] += (
                                torch.where(e1_image, float("-inf"), 0.0)
                                .unsqueeze(1).repeat(1, num_heads, 1)
                                .unsqueeze(-1).repeat(1, 1, 1, len(e2["prompt_indices"]))
                                .to(dtype=attn_weight.dtype)
                            )
                            attn_weight[:, :, num_text_tokens:, e1["prompt_indices"]] += (
                                torch.where(e2_image, float("-inf"), 0.0)
                                .unsqueeze(1).repeat(1, num_heads, 1)
                                .unsqueeze(-1).repeat(1, 1, 1, len(e1["prompt_indices"]))
                                .to(dtype=attn_weight.dtype)
                            )
                        if do_image_text_strengthening:
                            attn_weight[:, :, num_text_tokens:, e1["prompt_indices"]] *= (
                                torch.where(e1_image, k_strength, 1.0)
                                .unsqueeze(1).repeat(1, num_heads, 1)
                                .unsqueeze(-1).repeat(1, 1, 1, len(e1["prompt_indices"]))
                                .to(dtype=attn_weight.dtype)
                            )
                            attn_weight[:, :, num_text_tokens:, e2["prompt_indices"]] *= (
                                torch.where(e2_image, k_strength, 1.0)
                                .unsqueeze(1).repeat(1, num_heads, 1)
                                .unsqueeze(-1).repeat(1, 1, 1, len(e2["prompt_indices"]))
                                .to(dtype=attn_weight.dtype)
                            )
                        if do_text_text_weakening:
                            tt_shape = (
                                bs, num_heads, num_text_tokens, num_text_tokens,
                            )
                            t12 = torch.zeros(tt_shape, device=attn_weight.device, dtype=attn_weight.dtype)
                            t21 = torch.zeros_like(t12)
                            t12[:, :, e1["prompt_indices"][0], :] += 1
                            t12[:, :, :, e2["prompt_indices"][0]] += 1
                            t12 = torch.where(t12 > 1, float("-inf"), 0.0)
                            t21[:, :, e2["prompt_indices"][0], :] += 1
                            t21[:, :, :, e1["prompt_indices"][0]] += 1
                            t21 = torch.where(t21 > 1, float("-inf"), 0.0)
                            attn_weight[:, :, :num_text_tokens, :num_text_tokens] += t12
                            attn_weight[:, :, :num_text_tokens, :num_text_tokens] += t21

                        # ---- image-image: zero high cross-entity attention ----
                        if do_image_image:
                            n_img = attn_weight.shape[-1] - num_text_tokens
                            e1_q = e1_image.unsqueeze(-1).repeat(1, 1, n_img)
                            e2_q = e2_image.unsqueeze(-1).repeat(1, 1, n_img)
                            e1_k = e1_q.transpose(-1, -2)
                            e2_k = e2_q.transpose(-1, -2)
                            cross_12 = e1_q * e2_k
                            cross_21 = e2_q * e1_k

                            im_im = attn_weight_after_softmax[:, :, num_text_tokens:, num_text_tokens:].mean(dim=1)
                            for cross in (cross_12, cross_21):
                                weighted = (im_im * cross).reshape(bs, -1)
                                nz = weighted[weighted != 0]
                                if nz.numel() == 0:
                                    continue
                                threshold = nz.mean() + std_mul_image_image * nz.std()
                                mask = weighted > threshold
                                mask = torch.where(mask, float("-inf"), 0.0)
                                mask = mask.reshape(bs, n_img, n_img)
                                mask = mask.unsqueeze(1).repeat(1, num_heads, 1, 1).to(
                                    dtype=attn_weight.dtype, device=attn_weight.device
                                )
                                attn_weight[:, :, num_text_tokens:, num_text_tokens:] += mask

                        attn_weight_after_softmax = torch.softmax(attn_weight, dim=-1)

        hidden_states = attn_weight_after_softmax @ value
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return hidden_states

    @staticmethod
    def _scaled_dot_product_logits(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Scaled dot-product attention logits (pre-softmax).

        Mirrors ``F.scaled_dot_product_attention`` minus the softmax/dropout
        and the value matmul.
        """
        scale = 1 / math.sqrt(query.size(-1))
        return (query @ key.transpose(-2, -1)) * scale

    def _standard_forward(self, attn, hidden_states, encoder_hidden_states, image_rotary_emb):
        """Plain SDPA path used when deleaker is off."""
        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if encoder_hidden_states is not None:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            query = torch.cat([eq, query], dim=2)
            key = torch.cat([ek, key], dim=2)
            value = torch.cat([ev, value], dim=2)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return hidden_states


def _morphological_clean(mask_1d: torch.Tensor) -> np.ndarray:
    """Reshape a flat 0/1 image-token mask to its 2D grid, open + close it.

    Removes salt-and-pepper noise so the threshold-derived entity mask
    has connected blobs instead of speckle.
    """
    arr = mask_1d.detach().cpu().numpy().astype(np.uint8)
    side = int(round(np.sqrt(arr.shape[0])))
    if side * side != arr.shape[0]:
        # Not a square grid (shouldn't happen for FLUX), skip cleaning.
        return arr
    grid = (arr.reshape(side, side) * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    grid = cv2.morphologyEx(grid, cv2.MORPH_OPEN, kernel)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel, iterations=2)
    return (grid // 255).astype(np.uint8).reshape(-1)
