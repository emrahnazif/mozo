"""The nodes mozo ships.

Importing this package is what registers them: a node registers itself by being declared, so the
set of nodes is the set of modules named here. There is no list of node names for that set to
disagree with, and adding a node is adding a function.

Order matters only in that it is the editor's palette order.
"""

from __future__ import annotations

# The order is the editor's palette order, which is why it is not alphabetical: a workflow is
# built left to right, so the palette reads the same way -- where the pixels come from, what looks
# at them, what follows it from frame to frame, what changes them, what draws the answer on.
from . import io, model, track, transform, annotate  # noqa: F401,I001 -- imported to register them
