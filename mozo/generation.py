"""What a model that answers in words returns.

Every ``image_to_text`` family's ``predict`` returns a :class:`Generation`, and ``/predict`` sends
its :meth:`~Generation.to_dict` as JSON -- one type for the whole group, so one serving arm.

Not a plain ``str``, because of one field: ``finish_reason``. A generated answer is cut off when it
reaches the token budget, and a cut-off answer reads like a complete one more often than not. The
string alone cannot say which it is; this can. The two counts are what a caller needs to budget:
for an image-and-question model, the image is usually nearly all of the prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

__all__ = ["Generation"]


@dataclass(frozen=True)
class Generation:
    """A model's answer, in text.

    Attributes:
        text: The answer, decoded, with the model's control tokens removed.
        finish_reason: ``"stop"`` if the model ended the answer itself, ``"length"`` if it reached
            ``max_new_tokens`` first -- then the text is cut off, mid-sentence as likely as not.
        prompt_tokens: Tokens the model read, the image's included.
        generated_tokens: Tokens it wrote, the stopping token included.

    The text is the model's continuation, not a checked fact and not a probability: there is no
    confidence number, and it can be wrong with complete fluency.

    Examples:
        >>> g = Generation("A cat on a sofa.", "stop", 84, 6)
        >>> str(g)
        'A cat on a sofa.'
        >>> g.to_dict()["finish_reason"]
        'stop'
    """

    text: str
    finish_reason: Literal["stop", "length"]
    prompt_tokens: int
    generated_tokens: int

    def to_dict(self) -> dict:
        """The four fields, for JSON."""
        return asdict(self)

    def __str__(self) -> str:
        return self.text
