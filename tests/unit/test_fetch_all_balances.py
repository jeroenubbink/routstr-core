import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from routstr.wallet import fetch_all_balances


@asynccontextmanager
async def _fake_session():  # type: ignore[no-untyped-def]
    yield MagicMock()


class _TrackingSession:
    """Stand-in for ``AsyncSession`` that records concurrent use.

    Real ``AsyncSession`` is not safe for concurrent use: if a second
    coroutine issues a query while an ``await``-ed query is still in flight
    on the same session it raises ``"This session is provisioning a new
    connection; concurrent operations are not permitted"`` and can leave a
    connection wedged.

    Every awaited query method (``exec``, ``execute``, ``scalar``, ``get`` …)
    is resolved through ``__getattr__`` to the same tracked coroutine, so the
    double detects concurrent use regardless of which session API the code
    under test happens to call — it counts how many coroutines are inside a
    query at once and remembers the peak.
    """

    def __init__(self) -> None:
        self._in_flight = 0
        self.max_concurrency = 0
        self.query_calls = 0

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        async def _tracked(*args: object, **kwargs: object) -> MagicMock:
            self.query_calls += 1
            self._in_flight += 1
            self.max_concurrency = max(self.max_concurrency, self._in_flight)
            try:
                # Yield to the event loop so any concurrently-scheduled task
                # sharing this session can enter a query before we exit.
                await asyncio.sleep(0.01)
                result = MagicMock()
                result.one.return_value = 0
                result.all.return_value = []
                return result
            finally:
                self._in_flight -= 1

        return _tracked


def _tracking_session_factory() -> tuple[
    Callable[[], AbstractAsyncContextManager[_TrackingSession]],
    list[_TrackingSession],
]:
    """Patch replacement for ``db.create_session`` that records every session.

    Each ``async with db.create_session()`` gets a fresh ``_TrackingSession``,
    appended to ``sessions`` so the test can assert that *no single* session
    was ever used concurrently — true whether the fix precomputes balances in
    one query (one session, used serially) or opens a session per task.
    """
    sessions: list[_TrackingSession] = []

    @asynccontextmanager
    async def _factory() -> "AsyncIterator[_TrackingSession]":
        session = _TrackingSession()
        sessions.append(session)
        yield session

    return _factory, sessions


def _mint_patches(proof_amount: int = 1000):  # type: ignore[no-untyped-def]
    """Patches for the mint/proof side of ``fetch_all_balances`` (no DB)."""
    proof = MagicMock(amount=proof_amount)
    return [
        patch("routstr.wallet.get_wallet", AsyncMock(return_value=MagicMock())),
        patch(
            "routstr.wallet.get_proofs_per_mint_and_unit",
            MagicMock(return_value=[proof]),
        ),
        patch(
            "routstr.wallet.slow_filter_spend_proofs",
            AsyncMock(side_effect=lambda proofs, wallet: proofs),
        ),
    ]


def _patches(  # type: ignore[no-untyped-def]
    proof_amount: int = 1000, user_balance_msats: int = 0
):
    async def _grouped(  # type: ignore[no-untyped-def]
        session: object, mint_urls: list[str], units: list[str]
    ):
        return {(m, u): user_balance_msats for m in mint_urls for u in units}

    return _mint_patches(proof_amount) + [
        patch(
            "routstr.wallet.db.balances_by_mint_and_unit",
            AsyncMock(side_effect=_grouped),
        ),
        patch("routstr.wallet.db.create_session", _fake_session),
    ]


@pytest.mark.asyncio
async def test_fetch_all_balances_falls_back_to_primary_mint() -> None:
    """With empty cashu_mints, balances are still fetched for primary_mint."""
    from routstr.core.settings import settings

    with patch.object(settings, "cashu_mints", []), patch.object(
        settings, "primary_mint", "http://primary:3338"
    ):
        for p in _patches(proof_amount=1000):
            p.start()
        try:
            details, total_wallet, total_user, owner = await fetch_all_balances(
                units=["sat"]
            )
        finally:
            patch.stopall()

    assert [d["mint_url"] for d in details] == ["http://primary:3338"]
    assert total_wallet == 1000


@pytest.mark.asyncio
async def test_fetch_all_balances_reports_liability_when_wallet_is_empty() -> None:
    """An empty wallet must not hide outstanding user liabilities."""
    from routstr.core.settings import settings

    with patch.object(settings, "cashu_mints", []), patch.object(
        settings, "primary_mint", "http://primary:3338"
    ):
        for p in _patches(proof_amount=0, user_balance_msats=5000):
            p.start()
        try:
            details, total_wallet, total_user, owner = await fetch_all_balances(
                units=["sat"]
            )
        finally:
            patch.stopall()

    assert details[0]["wallet_balance"] == 0
    assert details[0]["user_balance"] == 5
    assert details[0]["owner_balance"] == -5
    assert total_wallet == 0
    assert total_user == 5
    assert owner == -5


@pytest.mark.asyncio
async def test_fetch_all_balances_does_not_use_one_session_concurrently() -> None:
    """A single AsyncSession must never be shared across the gathered tasks.

    ``fetch_all_balances`` fans out one balance lookup per (mint, unit) with
    ``asyncio.gather``. Passing one session into all of them makes concurrent
    coroutines issue queries on the same AsyncSession — unsafe, and the source
    of the observed "concurrent operations are not permitted" error that
    cascades into connection-pool exhaustion.

    We drive it with several mint/unit combinations and a session whose
    ``exec`` records peak concurrency. If any session is entered by more than
    one coroutine at a time the finding is present.
    """
    from routstr.core.settings import settings

    factory, sessions = _tracking_session_factory()

    # NB: db.balances_by_mint_and_unit is intentionally NOT patched here — we
    # let the real helper run so it issues a query on the session, which is
    # what exercises (and detects) concurrent session use.
    patches = _mint_patches() + [
        patch("routstr.wallet.db.create_session", factory),
    ]

    with patch.object(
        settings, "cashu_mints", ["http://mint-a:3338", "http://mint-b:3338"]
    ), patch.object(settings, "primary_mint", "http://mint-a:3338"):
        for p in patches:
            p.start()
        try:
            await fetch_all_balances(units=["sat", "msat"])
        finally:
            patch.stopall()

    # The liabilities must be read in exactly ONE short-lived session with ONE
    # query, before the concurrent mint fan-out — not a session per gathered
    # task, and never a session shared across tasks. Asserting the exact shape
    # (rather than a peak-concurrency counter, which cannot trip once no session
    # crosses the gather) makes the guarantee explicit and catches a regression
    # that reintroduces per-task DB access.
    assert len(sessions) == 1, (
        f"expected exactly one create_session() for the up-front liability read, "
        f"got {len(sessions)}"
    )
    assert sessions[0].query_calls == 1, (
        f"expected a single grouped liability query, got "
        f"{sessions[0].query_calls} — per-(mint,unit) querying has returned"
    )
    # And that single session was never entered concurrently.
    assert sessions[0].max_concurrency <= 1


@pytest.mark.asyncio
async def test_fetch_all_balances_degrades_when_liability_read_fails() -> None:
    """A DB failure reading liabilities must not 500 the whole balances page.

    A failed liability read does not invalidate custody, so the *known* wallet
    balance is still reported (both per-mint and in the wallet total). Only the
    unknowable per-user/owner split is blanked, and owner is reported as 0 so
    the custody is never claimed as owner profit.
    """
    from routstr.core.settings import settings

    patches = _mint_patches(proof_amount=1000) + [
        patch(
            "routstr.wallet.db.balances_by_mint_and_unit",
            AsyncMock(side_effect=RuntimeError("db pool exhausted")),
        ),
        patch("routstr.wallet.db.create_session", _fake_session),
    ]

    with patch.object(settings, "cashu_mints", []), patch.object(
        settings, "primary_mint", "http://primary:3338"
    ):
        for p in patches:
            p.start()
        try:
            details, total_wallet, total_user, owner = await fetch_all_balances(
                units=["sat"]
            )
        finally:
            patch.stopall()

    assert details[0]["error"] == "db pool exhausted"
    assert details[0]["user_balance"] == 0
    assert details[0]["owner_balance"] == 0
    # Custody is known and must not be hidden by the liability-read failure.
    assert details[0]["wallet_balance"] == 1000
    assert total_wallet == 1000
    # The user/owner split is unknown: blanked, and never claimed as profit.
    assert total_user == 0
    assert owner == 0


@pytest.mark.asyncio
async def test_liability_error_does_not_clobber_a_specific_mint_error() -> None:
    """A per-mint fetch failure keeps its specific error under a liability error.

    When the up-front liability read fails, every detail is tagged so the UI
    shows liabilities are unknown — but a detail that already failed at the mint
    level carries a more specific message, which must not be overwritten by the
    generic liability-read error.
    """
    from routstr.core.settings import settings

    patches: list[Any] = [
        patch(
            "routstr.wallet.get_wallet",
            AsyncMock(side_effect=RuntimeError("mint down")),
        ),
        patch(
            "routstr.wallet.db.balances_by_mint_and_unit",
            AsyncMock(side_effect=RuntimeError("db pool exhausted")),
        ),
        patch("routstr.wallet.db.create_session", _fake_session),
    ]

    with patch.object(settings, "cashu_mints", []), patch.object(
        settings, "primary_mint", "http://primary:3338"
    ):
        for p in patches:
            p.start()
        try:
            details, *_ = await fetch_all_balances(units=["sat"])
        finally:
            patch.stopall()

    assert details[0]["error"] == "mint down"


@pytest.mark.asyncio
async def test_fetch_all_balances_no_duplicate_primary_mint() -> None:
    """primary_mint already in cashu_mints is not inspected twice."""
    from routstr.core.settings import settings

    with patch.object(
        settings, "cashu_mints", ["http://primary:3338"]
    ), patch.object(settings, "primary_mint", "http://primary:3338"):
        for p in _patches(proof_amount=1000):
            p.start()
        try:
            details, total_wallet, _total_user, _owner = await fetch_all_balances(
                units=["sat"]
            )
        finally:
            patch.stopall()

    assert [d["mint_url"] for d in details] == ["http://primary:3338"]
    assert total_wallet == 1000
