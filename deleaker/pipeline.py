"""Pipeline that wraps diffusers' ``FluxPipeline`` with the deleaker intervention.

Subclasses ``FluxPipeline`` so users can call ``DeleakerFluxPipeline.from_pretrained(...)``
exactly like the stock pipeline. The only API additions are the ``entities``,
``use_deleaker`` and ``deleaker_config`` kwargs to ``__call__``.
"""

from typing import List, Optional

import torch
from diffusers import FluxPipeline

from .attention_processor import DeleakerFluxAttnProcessor2_0
from .attention_store import AttentionStore
from .config import DeleakerConfig
from .tokens import find_entity_token_indices


def _register_deleaker_processors(transformer, attention_store: AttentionStore) -> None:
    """Replace every ``transformer_blocks.*`` and ``single_transformer_blocks.*``
    attention processor with a ``DeleakerFluxAttnProcessor2_0``.
    """
    procs = {}
    for name in transformer.attn_processors.keys():
        layer_name = ".".join(name.split(".")[:2])
        procs[name] = DeleakerFluxAttnProcessor2_0(
            layer_name=layer_name, attention_store=attention_store
        )
    transformer.set_attn_processor(procs)


class DeleakerFluxPipeline(FluxPipeline):
    """``FluxPipeline`` with optional deleaker attention rewriting.

    Example
    -------
    >>> pipe = DeleakerFluxPipeline.from_pretrained(
    ...     "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
    ... ).to("cuda")
    >>> img = pipe(
    ...     prompt="A cat and a cheetah are splashing each other",
    ...     entities=["cat", "cheetah"],
    ...     num_inference_steps=20, guidance_scale=3.5,
    ... ).images[0]
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        entities: Optional[List[str]] = None,
        use_deleaker: bool = True,
        deleaker_config: Optional[DeleakerConfig] = None,
        **kwargs,
    ):
        # Pure-vanilla path: no entities given OR disabled. Delegate straight
        # to the stock pipeline.
        if entities is None or not use_deleaker:
            return super().__call__(prompt=prompt, **kwargs)

        if isinstance(prompt, list):
            raise ValueError(
                "Deleaker only supports a single prompt at a time; "
                "got a list of prompts."
            )

        cfg = deleaker_config or DeleakerConfig()
        token_indices = find_entity_token_indices(self.tokenizer_2, prompt, entities)

        store = AttentionStore()
        _register_deleaker_processors(self.transformer, store)

        # Shared dict — the pre-forward hook mutates ``step_index`` before each
        # call into ``self.transformer``, and the attention processor reads it
        # via ``joint_attention_kwargs["deleaker_kwargs"]``.
        deleaker_kwargs = cfg.to_dict()
        deleaker_kwargs["use_deleaker"] = use_deleaker
        deleaker_kwargs["token_entity_indices"] = token_indices
        deleaker_kwargs["step_index"] = 0

        counter = {"i": 0}

        def _pre_hook(module, args, kwargs_):
            deleaker_kwargs["step_index"] = counter["i"]
            counter["i"] += 1

        hook_handle = self.transformer.register_forward_pre_hook(
            _pre_hook, with_kwargs=True
        )

        jak = dict(kwargs.pop("joint_attention_kwargs", None) or {})
        jak["deleaker_kwargs"] = deleaker_kwargs

        try:
            return super().__call__(prompt=prompt, joint_attention_kwargs=jak, **kwargs)
        finally:
            hook_handle.remove()
