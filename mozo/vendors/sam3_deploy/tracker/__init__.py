# SPDX-License-Identifier: Apache-2.0
"""SAM 3's video tracker: concepts or clicked objects followed through a stream of frames.

Derived from ``transformers/models/sam3_tracker_video`` and ``transformers/models/sam3_video``
(Apache-2.0), with Meta's published model as the reference it is checked against. See
``PROVENANCE.md``.
"""

from .session import Frame, VideoTracker

__all__ = ["Frame", "VideoTracker"]
