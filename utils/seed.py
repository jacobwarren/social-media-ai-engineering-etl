import random

try:
    import numpy as _np
except Exception:
    _np = None

try:
    import torch as _torch
except Exception:
    _torch = None


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    try:
        random.seed(seed)
    except Exception:
        pass
    if _np is not None:
        try:
            _np.random.seed(seed)
        except Exception:
            pass
    if _torch is not None:
        try:
            _torch.manual_seed(seed)
            if hasattr(_torch, "cuda") and _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

