from typing import Optional, List, Dict, Any

try:
    from pydantic import BaseModel, Field
    _PYDANTIC = True
except Exception:
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore
    _PYDANTIC = False


class PostRecord(BaseModel):  # type: ignore[misc]
    post_text: Optional[str] = None
    author_name: Optional[str] = None
    is_repost: Optional[bool] = None
    structure: Optional[str] = None
    topic: Optional[str] = None
    opinion: Optional[str] = None
    context: Optional[str] = None
    tone: Optional[str] = None


class StyleFeaturesRecord(PostRecord):  # type: ignore[misc]
    sentence_structure: Optional[List[int]] = None
    vocabulary_usage: Optional[int] = None
    line_breaks: Optional[int] = None
    avg_line_breaks: Optional[float] = None
    punctuation_usage: Optional[Dict[str, int]] = None
    divider_style: Optional[str] = None
    bullet_styles: Optional[str] = None
    topic_shifts: Optional[List[Dict[str, Any]]] = None
    flow: Optional[List[str]] = None
    pacing: Optional[str] = None
    sentiment_arc: Optional[str] = None
    profanity: Optional[str] = None
    common_phrases: Optional[List[str]] = None


class PromptRecord(StyleFeaturesRecord):  # type: ignore[misc]
    prompt: Optional[str] = None


# Lightweight "validate" that works even if pydantic is missing

def validate_record(model_cls, data: Dict[str, Any]) -> Dict[str, Any]:
    if _PYDANTIC and issubclass(model_cls, BaseModel):
        return model_cls(**data).dict()  # type: ignore[attr-defined]
    # Best-effort: return data unchanged
    return data



# Standardized artifact shape hints
class Stage01Record(BaseModel):  # type: ignore[misc]
    post_text: Optional[str] = None
    tier: Optional[str] = None
    engagement_ratio: Optional[float] = None


class Stage03Record(PostRecord):  # type: ignore[misc]
    structure: Optional[str] = None


class Stage17Record(StyleFeaturesRecord):  # type: ignore[misc]
    pass


class Stage18Record(PromptRecord):  # type: ignore[misc]
    pass


class Stage22Row(BaseModel):  # type: ignore[misc]
    system: Optional[str] = None
    prompt: Optional[str] = None
    chosen: Optional[str] = None
    rejected: Optional[str] = None


class Stage23Row(Stage22Row):  # type: ignore[misc]
    pass
