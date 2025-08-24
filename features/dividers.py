from __future__ import annotations

from collections import Counter
from typing import Optional

from .context import FeatureContext


def detect_divider_styles(text: str, ctx: FeatureContext) -> Optional[str]:
    lines = text.split('\n')
    dividers = []
    for line in lines:
        match = ctx.divider_re.match(line)
        if match:
            dividers.append(match.group(1))
    divider_counts = Counter(dividers)
    return divider_counts.most_common(1)[0][0] if divider_counts else None

