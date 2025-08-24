from __future__ import annotations
from typing import List
import re


def punctuation_usage_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    scores: List[float] = []
    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]
        m = re.search(r"\*\*Punctuation\*\*:\s*(.*?)(?:\n|$)", user_prompt)
        requested_style = None
        if m:
            txt = m.group(1).lower()
            requested_style = {
                "periods": 2 if "heavy use of periods" in txt else 1,
                "commas": 2 if "heavy use of commas" in txt else 1,
                "exclamation": 2 if "heavy use of exclamation" in txt else 1,
                "question": 2 if "heavy use of question" in txt else 1,
                "semicolon": 2 if "heavy use of semicolons" in txt else 1,
            }
        raw_text = completion if isinstance(completion, str) else completion[0]
        answer_text = raw_text if raw_text.strip() else raw_text
        if not requested_style:
            scores.append(0.5)
            continue
        text_length = len(answer_text)
        if text_length == 0:
            scores.append(0.3)
            continue
        punct_counts = {
            "periods": answer_text.count('.') / text_length,
            "commas": answer_text.count(',') / text_length,
            "exclamation": answer_text.count('!') / text_length,
            "question": answer_text.count('?') / text_length,
            "semicolon": answer_text.count(';') / text_length,
        }
        thresholds = {
            "periods": {"low": 0.01, "normal": 0.02, "heavy": 0.03},
            "commas": {"low": 0.01, "normal": 0.02, "heavy": 0.03},
            "exclamation": {"low": 0.001, "normal": 0.005, "heavy": 0.01},
            "question": {"low": 0.001, "normal": 0.005, "heavy": 0.01},
            "semicolon": {"low": 0.0005, "normal": 0.001, "heavy": 0.002},
        }
        type_scores = []
        for punct_type, desired_level in requested_style.items():
            actual_freq = punct_counts[punct_type]
            if desired_level == 2:
                if actual_freq >= thresholds[punct_type]["heavy"]:
                    type_scores.append(1.0)
                elif actual_freq >= thresholds[punct_type]["normal"]:
                    type_scores.append(0.7)
                elif actual_freq >= thresholds[punct_type]["low"]:
                    type_scores.append(0.4)
                else:
                    type_scores.append(0.1)
            else:
                if thresholds[punct_type]["low"] <= actual_freq <= thresholds[punct_type]["normal"]:
                    type_scores.append(1.0)
                elif actual_freq < thresholds[punct_type]["low"]:
                    type_scores.append(0.6)
                elif actual_freq < thresholds[punct_type]["heavy"]:
                    type_scores.append(0.8)
                else:
                    type_scores.append(0.4)
        scores.append(sum(type_scores) / len(type_scores))
    return scores

