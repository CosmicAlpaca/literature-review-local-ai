from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import guardrails


@pytest.fixture(autouse=True)
def reset_in_process_rate_limiter():
    """Keep the production rate limiter enabled while isolating test state."""
    with guardrails._rate_lock:
        guardrails._rate_state.clear()
    yield
    with guardrails._rate_lock:
        guardrails._rate_state.clear()
