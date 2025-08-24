from __future__ import annotations

from collections import Counter
from typing import Optional
from .context import FeatureContext


def detect_bullet_styles(text: str, ctx: FeatureContext) -> Optional[str]:
    lines = text.split('\n')
    bullet_list = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if ctx.bul_numbered_re.match(line):
            bullet_list.append("Numbered"); continue
        if ctx.bul_lettered_re.match(line):
            bullet_list.append("Lettered"); continue
        indent_bullet = ctx.bul_indent_re.match(line)
        if indent_bullet:
            bullet_list.append(indent_bullet.group(1)); continue
        sym_match = ctx.bul_symbolic_re.match(line)
        if sym_match:
            bullet_list.append(sym_match.group(1)); continue
        first_word = line.split()[0] if line.split() else ''
        # crude check: if first word is entirely emojis
        try:
            import emojis
            if first_word and all(emojis.count(ch) for ch in first_word):
                bullet_list.append("EmojiBullets" if len(first_word) > 1 else "Emoji")
        except Exception:
            pass
    if not bullet_list:
        return None
    counts = Counter(bullet_list)
    emoji_entries = [b for b in counts if "Emoji" in b]
    if len(emoji_entries) > 1:
        return "Differing Emojis"
    if len(counts) > 1:
        return "Mixed Bullet Styles"
    style, _ = counts.most_common(1)[0]
    return style

