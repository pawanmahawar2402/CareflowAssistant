"""Shared test setup: synthetic data, and a Redis-backed marker that skips memory tests when Redis is absent."""
import runpy
from pathlib import Path

import pytest

from careflow.config import settings

ROOT = Path(__file__).parent.parent


def _redis_up() -> bool:
    try:
        import redis

        return bool(redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1).ping())
    except Exception:
        return False


requires_redis = pytest.mark.skipif(
    not _redis_up(), reason="needs Redis: docker run -d -p 6379:6379 redis:7-alpine")


@pytest.fixture(scope="session", autouse=True)
def synthetic_data():
    runpy.run_path(str(ROOT / "generate_data.py"))
    settings.careflow_data_dir = str(ROOT / "data")
