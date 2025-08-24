import random
import re
from typing import Dict, Any

EMOJI = "😀😃😄😁😆😅😂😊🙂🙃😉😍😘😜🤔🤩😎🥳😭🤯🔥✨👍👎👉👇✅❌💡📈📉"


def violate_length(text: str, target: str = "over", max_over_ratio: float = 0.2) -> str:
    if target == "under":
        return text[: max(1, int(len(text) * 0.5))]
    # over: append filler to exceed
    filler = " Lorem ipsum dolor sit amet." * max(1, int(len(text) * max_over_ratio / 24))
    return text + filler


def violate_emoji(text: str, severity: str = "high") -> str:
    if severity == "none":
        return re.sub(r"[\U00010000-\U0010FFFF]", "", text)
    # Inject many emojis
    tail = "".join(random.choice(EMOJI) for _ in range(10 if severity == "high" else 4))
    return text + "\n\n" + tail


def violate_hashtags(text: str, count: int = 8) -> str:
    tags = [f"#tag{i}" for i in range(count)]
    return text.rstrip() + "\n\n" + " ".join(tags)


def violate_urls(text: str) -> str:
    url = "https://lnkd.in/" + "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(8))
    return text + f"\n\nLearn more: {url}"


def violate_names(text: str) -> str:
    name = random.choice(["John Doe", "Jane Smith", "Alex Johnson", "Chris Lee"]) 
    return text + f"\n\nShout out to {name}!"


def generate_negative(chosen: str, constraints: Dict[str, Any]) -> str:
    """Generate a negative by violating one or more constraints.
    constraints can include: length ('up to 750', 'between 750 and 1500'),
    emoji_usage ('none'|'infrequent'|'frequent'), hashtag_limit (<=3), allow_urls (False), allow_names (False).
    """
    text = chosen
    # Choose violations based on constraints
    if constraints.get("allow_urls") is False:
        text = violate_urls(text)
    if constraints.get("allow_names") is False:
        text = violate_names(text)
    usage = (constraints.get("emoji_usage") or "").lower()
    if usage in ["none", "infrequent"]:
        text = violate_emoji(text, severity="high")
    # hashtag limit
    limit = constraints.get("hashtag_limit", 3)
    text = violate_hashtags(text, count=max(6, limit + 3))
    # length
    length_req = (constraints.get("length") or "").lower()
    if "up to" in length_req:
        text = violate_length(text, target="over")
    elif "between" in length_req:
        text = violate_length(text, target="under")
    return text

