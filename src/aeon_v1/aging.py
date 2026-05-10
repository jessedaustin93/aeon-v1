from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


def age_weight(created_at: str, config: "Config", importance: float = 0.0) -> float:
    """Return a recency weight in [min_weight, 1.0] for a memory.

    Uses exponential half-life decay so the weight halves every
    `memory_aging_half_life_days` days.  The effective half-life is stretched
    by the memory's `importance` score:
        effective_half_life = half_life * (1 + importance * importance_scale)
    So a 1.0-importance memory with scale=1.0 decays at half the rate of a
    0.0-importance memory — it stays "front and center" roughly twice as long.

    Very old memories floor at `memory_aging_min_weight` so they never vanish.
    When `memory_aging_enabled` is False the function always returns 1.0.
    """
    if not getattr(config, "memory_aging_enabled", True):
        return 1.0

    if not created_at:
        return 1.0

    _hl = getattr(config, "memory_aging_half_life_days", 30.0)
    half_life = float(_hl if _hl is not None else 30.0)
    _mw = getattr(config, "memory_aging_min_weight", 0.2)
    min_weight = float(_mw if _mw is not None else 0.2)
    min_weight = max(0.0, min(1.0, min_weight))
    _is = getattr(config, "memory_aging_importance_scale", 1.0)
    importance_scale = float(_is if _is is not None else 1.0)

    importance = max(0.0, min(1.0, float(importance)))
    effective_half_life = half_life * (1.0 + importance * importance_scale)

    try:
        ts = created_at.strip()
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        elif "+" not in ts and len(ts) >= 19:
            ts = ts + "+00:00"
        created_dt = datetime.fromisoformat(ts[:26])
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return 1.0

    now = datetime.now(timezone.utc)
    age_days = max(0.0, (now - created_dt).total_seconds() / 86400.0)
    weight = 0.5 ** (age_days / effective_half_life)
    return max(min_weight, weight)
