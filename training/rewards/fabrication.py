from __future__ import annotations
from typing import List
import re

from training.grpo.prompt_parsing import (
    extract_prompt_content,
    detect_urls,
    detect_potential_people_names,
    detect_organization_names,
)


def fabrication_detection_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Penalize fabricated content that is not supported by the prompt content.
    Returns per-sample rewards in [0, 1], where 1.0 means no fabrication detected.
    Extracted from the original root 26-train-grpo.py implementation.
    """
    scores: List[float] = []
    for i, completion in enumerate(completions):
        user_prompt = prompts[i][-1] if isinstance(prompts[i], list) else prompts[i]
        raw_text = completion if isinstance(completion, str) else (completion[0] if completion else "")
        answer_text = raw_text if raw_text.strip() else raw_text

        # Extract prompt content
        prompt_content = extract_prompt_content(user_prompt)

        fabrication_penalty = 0.0

        # URLs introduced that weren't present in prompt/key message
        urls_in_completion = detect_urls(answer_text)
        urls_in_prompt = detect_urls(prompt_content.get('full_text', ''))
        urls_in_key_message = detect_urls(prompt_content.get('key_message', ''))
        if urls_in_completion:
            if not urls_in_prompt and not urls_in_key_message:
                fabrication_penalty += 0.7
            else:
                if not any(u_c == u_p for u_c in urls_in_completion for u_p in (urls_in_prompt + urls_in_key_message)):
                    fabrication_penalty += 0.5

        # New people names
        names_in_completion = detect_potential_people_names(answer_text)
        names_in_topic = detect_potential_people_names(prompt_content.get('topic', ''))
        names_in_key_message = detect_potential_people_names(prompt_content.get('key_message', ''))
        names_in_prompt = set([n.lower() for n in (names_in_topic + names_in_key_message)])
        if names_in_completion:
            new_names = 0
            for nm in names_in_completion:
                if nm and nm.lower() not in names_in_prompt:
                    new_names += 1
            if new_names > 0:
                fabrication_penalty += min(0.5, new_names * 0.1)

        # New organization names
        orgs_in_completion = detect_organization_names(answer_text)
        orgs_in_topic = detect_organization_names(prompt_content.get('topic', ''))
        orgs_in_key_message = detect_organization_names(prompt_content.get('key_message', ''))
        orgs_in_prompt = set([o.lower() for o in (orgs_in_topic + orgs_in_key_message)])
        if orgs_in_completion:
            new_orgs = 0
            for org in orgs_in_completion:
                if org and org.lower() not in orgs_in_prompt:
                    new_orgs += 1
            if new_orgs > 0:
                fabrication_penalty += min(0.5, new_orgs * 0.1)

        # Promotional phrases not present in prompt
        newsletter_patterns = [
            r"(sign\s*up|subscribe|join).{0,30}(newsletter)",
            r"(register|join).{0,30}(webinar|event)",
            r"link in (bio|profile|comments)",
        ]
        for pattern in newsletter_patterns:
            if (re.search(pattern, answer_text, flags=re.IGNORECASE)
                and not re.search(pattern, prompt_content.get('topic', ''), flags=re.IGNORECASE)
                and not re.search(pattern, prompt_content.get('key_message', ''), flags=re.IGNORECASE)):
                fabrication_penalty += 0.3
                break

        # Action-driving phrases not present in prompt
        action_patterns = [
            r"link in (bio|comments|description)",
            r"dm me for",
            r"email me at",
            r"call (me|us) at",
            r"limited time offer",
            r"exclusive (deal|offer)",
        ]
        for pattern in action_patterns:
            if (re.search(pattern, answer_text, flags=re.IGNORECASE)
                and not re.search(pattern, prompt_content.get('topic', ''), flags=re.IGNORECASE)
                and not re.search(pattern, prompt_content.get('key_message', ''), flags=re.IGNORECASE)):
                fabrication_penalty += 0.4
                break

        fabrication_score = max(0.0, 1.0 - fabrication_penalty)
        scores.append(fabrication_score)

    return scores

