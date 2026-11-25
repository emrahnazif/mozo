# SPDX-License-Identifier: Apache-2.0
"""PP-OCRv6, extracted for deployment.

Text detection and recognition in two graphs: a DB detector finds the lines, an SVTR/CTC
recogniser reads them. See PROVENANCE.md for what this was taken from and what changed.

    >>> from mozo.vendors.ppocrv6_deploy import Reader, SPECS        # doctest: +SKIP
    >>> reader = Reader("torch-fp32.pth", SPECS["medium"])            # doctest: +SKIP
    >>> [(r.text, round(r.confidence, 2)) for r in reader(image)]     # doctest: +SKIP
    [('EXIT 42', 1.0)]
"""

from .config import SPECS, VARIANTS, Spec
from .predictor import Reader, Region

__all__ = ["Reader", "Region", "SPECS", "Spec", "VARIANTS"]
