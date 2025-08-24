from typing import List, Optional

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import pandera as pa
    from pandera.typing import DataFrame
except Exception:  # pragma: no cover
    pa = None  # type: ignore
    DataFrame = None  # type: ignore


class BasicSchema:  # lightweight default schema
    required_columns = ["system", "prompt", "chosen", "rejected"]


# Optional stricter schema with Pandera; still safe without it
if pa is not None:
    class StrictPairsSchema(pa.SchemaModel):  # type: ignore
        system: pa.typing.Series[str]
        prompt: pa.typing.Series[str]
        chosen: pa.typing.Series[str]
        rejected: pa.typing.Series[object]

        class Config:
            coerce = True


def validate_csv(path: str, required_columns: Optional[List[str]] = None) -> bool:
    """Validate a CSV has the required columns and at least one row.

    Returns True if basic validation passes, False otherwise. If pandera is
    available, we apply a stricter schema to catch type issues.
    """
    if pd is None:
        return True
    try:
        df = pd.read_csv(path)
        cols = set(df.columns)
        req = set(required_columns or BasicSchema.required_columns)
        if not req.issubset(cols):
            return False
        if len(df) < 1:
            return False
        if pa is not None:
            try:
                StrictPairsSchema.validate(df)  # type: ignore[name-defined]
            except Exception:
                return False
        return True
    except Exception:
        return False




def validate_jsonl_records(path: str,
                            model_cls=None,
                            required_keys: Optional[List[str]] = None,
                            allowed_values: Optional[dict] = None,
                            sample_limit: Optional[int] = None) -> bool:
    """Validate a JSONL file by checking required keys and optional allowed values.

    If model_cls is provided and looks like a Pydantic model, we instantiate it
    per-record to enforce types. Otherwise, we just check keys and simple enums.
    Returns True on success, False otherwise.
    """
    try:
        import json
        n = 0
        ok = True
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                n += 1
                rec = json.loads(line)
                if required_keys:
                    for k in required_keys:
                        if k not in rec:
                            return False
                if allowed_values:
                    for k, allowed in allowed_values.items():
                        if k in rec and rec[k] not in allowed:
                            return False
                if model_cls is not None:
                    try:
                        # If using Pydantic-style class, instantiate it
                        model_cls(**rec)  # type: ignore[misc]
                    except Exception:
                        return False
                if sample_limit and n >= sample_limit:
                    break
        return n > 0 and ok
    except Exception:
        return False
