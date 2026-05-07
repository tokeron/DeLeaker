"""Map each entity string to the token positions it occupies in the prompt."""

from typing import Iterable


def find_entity_token_indices(tokenizer, prompt: str, entities: Iterable[str]) -> dict:
    """Return ``{entity: range(start, end)}`` of token positions in ``prompt``.

    Tries the entity as-given, then lowercased, then prefixed with a space.
    Raises ``ValueError`` if any entity cannot be located.
    """
    tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
    indices: dict = {}

    def strip_special(token_ids):
        first = tokenizer.decode(token_ids[0])
        if first in ("", "<bos>"):
            return token_ids[1:]
        return token_ids

    for entity in entities:
        for variant in (entity, entity.lower(), " " + entity):
            tok = tokenizer(variant, return_tensors="pt", padding=False, truncation=True)
            ids = strip_special(tok["input_ids"][0])
            if len(ids) == 0:
                continue
            target = ids[:-1] if len(ids) > 1 else ids
            n = len(target)
            for i in range(len(tokenized_prompt["input_ids"][0]) - n + 1):
                if (tokenized_prompt["input_ids"][0][i:i + n] == target).all():
                    indices[entity] = range(i, i + n)
                    break
            if entity in indices:
                break
        if entity not in indices:
            raise ValueError(
                f"Entity {entity!r} not found in prompt {prompt!r}. "
                "Make sure each entity appears verbatim in the prompt."
            )
    return indices
