from __future__ import annotations

from typing import Literal


def determine_profanity_category(text: str) -> Literal["none","light","moderate","heavy"]:
    categorized_words = {
        "apeshit": "moderate",
        "arsehole": "light",
        "ass": "light",
        "asshole": "light",
        "bastard": "moderate",
        "bullshit": "moderate",
        "bitch": "moderate",
        "clusterfuck": "heavy",
        "damn": "moderate",
        "damnit": "moderate",
        "fuck": "heavy",
        "fucker": "heavy",
        "fuckin": "heavy",
        "fucking": "heavy",
        "goddamn": "heavy",
        "bollocks": "light",
        "hell": "light",
        "holy shit": "moderate",
        "horseshit": "moderate",
        "motherfucker": "heavy",
        "mother fucker": "heavy",
        "piss": "light",
        "pissed": "light",
        "shit": "moderate",
    }
    words_in_text = text.lower().split()
    highest_severity = "none"
    severity_mapping = {"none": 0, "light": 1, "moderate": 2, "heavy": 3}
    for word in words_in_text[:1000]:
        category = categorized_words.get(word, "none")
        if severity_mapping[category] > severity_mapping[highest_severity]:
            highest_severity = category
    return highest_severity

