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
    """Several strings -- class names, keypoint names -- as a list, each exactly as given.

    A list is taken as it is, a comma inside an entry included: ``"bolt, M8"`` is one name, not
    two. Empty entries are dropped. A string is the old form, from workflows saved before these
    were lists, and keeps its old meaning: split by :func:`comma_separated`. The editor splits one
    the same way when it shows it, so editing it keeps what it meant. Prompts, whose positions
    matter, are read with :func:`prompt_list` instead.
    """
    if value is None or isinstance(value, str):
        return comma_separated(value)
    return [str(prompt).strip() for prompt in value if str(prompt).strip()]


def prompt_list(value: Union[str, Sequence[str], None]) -> list:
    """Prompts, as :func:`text_list` reads them -- except that a list's empty entry is refused.

    Each result's ``class_id`` is its prompt's position (see :data:`mozo.workflow.node.Prompts`),
    so dropping an empty entry would silently move every prompt after it. The old comma string is
    still split as :func:`text_list` splits it: the positions a person sees in it are those left.

    Raises:
        ValueError: If a list has an entry that is empty, or only whitespace.
    """
    if value is not None and not isinstance(value, str):
        empty = [index for index, prompt in enumerate(value) if not str(prompt).strip()]
        if empty:
            raise ValueError(
                f"prompt {empty[0]} of {len(value)} is empty. Each result's class_id is its "
                f"prompt's position, so dropping it would move every prompt after it; remove it, "
                f"or give it text.")
    return text_list(value)
