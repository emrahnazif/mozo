"""What a person typed into a text box, as a list.

One rule -- split on commas, strip, drop the empties -- and one home, because it is the contract for
every field where several things go in one box: the phrases a zero-shot classifier scores, the
concepts an open-vocabulary detector looks for, the class names a prediction is filtered to, the
keypoints a box is redrawn around. It was written three times before this existed, in two packages,
and widening it (a semicolon, a newline) would have meant finding all three.

Commas rather than Grounding DINO's upstream ``"person . car . dog"``: mozo's adapters refuse a
prompt containing ``.``, and one separator everywhere is one thing to learn instead of four.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union


def comma_separated(text: Optional[str]) -> list:
    """Split *text* on commas. Unset or empty gives an empty list."""
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def text_list(value: Union[str, Sequence[str], None]) -> list:
    """Several strings -- prompts, class names -- as a list, each exactly as given.

    A list is taken as it is, a comma inside an entry included, because a prompt is free text and
    ``"bolt, M8"`` is one concept, not two. Splitting it would hand every later prompt the wrong
    ``class_id``: results that look right and are not. A string is the old form, from workflows
    saved before these were lists, and keeps its old meaning: split by :func:`comma_separated`.
    The editor splits one the same way when it shows it, so editing it keeps what it meant.
    """
    if value is None or isinstance(value, str):
        return comma_separated(value)
    return [str(prompt).strip() for prompt in value if str(prompt).strip()]
