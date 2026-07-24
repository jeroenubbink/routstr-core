"""Coverage for env-configurable DB connection-pool sizing.

Pool sizing comes from the pydantic ``Settings`` (DATABASE_POOL_SIZE /
DATABASE_MAX_OVERFLOW / DATABASE_POOL_TIMEOUT), matching how every other typed
env var is consumed. Defaults equal SQLAlchemy's own baseline so unset is
behaviour-neutral; a bad value fails validation and refuses to boot (like a
malformed DATABASE_URL); the effective config is logged at engine creation; and
pool_pre_ping is never enabled (the default DB is a local SQLite file with no
network peer to drop idle connections). In-memory SQLite uses StaticPool, which
rejects the pool kwargs, so those URLs are built without them.
"""

import logging

import pytest
from pydantic.v1 import ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.pool import QueuePool, StaticPool

from routstr.core.db import create_db_engine
from routstr.core.settings import Settings, settings

# A file URL keeps the pool an AsyncAdaptedQueuePool (which honours pool_size);
# engine creation is lazy, so no connection is opened and no file is created.
_URL = "sqlite+aiosqlite:///./_pool_config_test.db"


def _pool(engine: AsyncEngine) -> QueuePool:
    pool = engine.sync_engine.pool
    assert isinstance(pool, QueuePool)
    return pool


def test_pool_defaults_match_sqlalchemy_baseline() -> None:
    """With defaults, the pool keeps SQLAlchemy's 5 / 10 / 30 baseline."""
    pool = _pool(create_db_engine(_URL))
    assert pool.size() == 5
    assert pool._max_overflow == 10
    assert pool._timeout == 30


def test_pool_config_reads_settings_overrides() -> None:
    """create_db_engine sizes the pool from the Settings values."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(settings, "database_pool_size", 3)
        mp.setattr(settings, "database_max_overflow", 7)
        mp.setattr(settings, "database_pool_timeout", 12)

        pool = _pool(create_db_engine(_URL))

    assert pool.size() == 3
    assert pool._max_overflow == 7
    assert pool._timeout == 12


@pytest.mark.parametrize("url", ["sqlite+aiosqlite://", "sqlite+aiosqlite:///:memory:"])
def test_in_memory_sqlite_skips_pool_sizing(url: str) -> None:
    """In-memory SQLite uses StaticPool, which rejects pool kwargs.

    Passing pool_size/max_overflow/pool_timeout to such a URL raises TypeError,
    so create_db_engine must not send them — the engine must build cleanly.
    """
    engine = create_db_engine(url)
    assert isinstance(engine.sync_engine.pool, StaticPool)


def test_pool_does_not_enable_pre_ping() -> None:
    """pre_ping stays off: the local SQLite file has no peer to drop connections."""
    assert _pool(create_db_engine(_URL))._pre_ping is False


def test_engine_creation_logs_effective_pool_config(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The effective pool config is logged so operators can confirm it at boot."""
    # The routstr loggers set propagate=False, so caplog's root handler never
    # sees the record — attach the capture handler to this logger directly.
    db_logger = logging.getLogger("routstr.core.db")
    db_logger.addHandler(caplog.handler)
    db_logger.setLevel(logging.INFO)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(settings, "database_pool_size", 4)
            mp.setattr(settings, "database_max_overflow", 9)
            mp.setattr(settings, "database_pool_timeout", 15)
            create_db_engine(_URL)
    finally:
        db_logger.removeHandler(caplog.handler)

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "pool_size" in logged
    assert "4" in logged and "9" in logged and "15" in logged


# --- Validation happens in Settings (pydantic), like every other typed env var.
# A bad value raises ValidationError when Settings is built at boot, refusing to
# start rather than running misconfigured.


def test_non_integer_pool_size_rejected_at_boot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_POOL_SIZE", "twenty")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("value", ["0", "-3"])
def test_out_of_range_pool_size_rejected_at_boot(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """pool_size below 1 is rejected: 0 would mean an unbounded pool."""
    monkeypatch.setenv("DATABASE_POOL_SIZE", value)
    with pytest.raises(ValidationError):
        Settings()


def test_negative_pool_timeout_rejected_at_boot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_POOL_TIMEOUT", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_zero_max_overflow_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_overflow=0 (no overflow) is a valid choice, not a rejection trigger."""
    monkeypatch.setenv("DATABASE_MAX_OVERFLOW", "0")
    assert Settings().database_max_overflow == 0
