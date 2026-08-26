"""How a model write decides the provenance it stores.

``manual`` means "an operator set this price", so it is claimed on a price edit
and on nothing else. Any other edit — a rename, an enable, a context change, a
batch "save as fetched" — must leave the resolved source intact, or the first
catalog refresh would relabel every model as operator-entered and destroy the
provenance this exists to record.

Two comparisons make that hold, and both are easy to get wrong: the price is
compared against the same fee-free view the operator was shown (which carries
cache rates the stored JSON never had), and a payload is canonicalised before
it is compared, because the write replaces the whole stored price and a rate it
omits really does drop to zero.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from pydantic.v1 import ValidationError
from sqlmodel.ext.asyncio.session import AsyncSession

from routstr.core.admin import admin_sessions
from routstr.core.db import ModelRow, UpstreamProviderRow
from routstr.payment.models import _row_to_model
from routstr.proxy import reinitialize_upstreams

_ARCHITECTURE = {
    "modality": "text",
    "input_modalities": ["text"],
    "output_modalities": ["text"],
    "tokenizer": "unknown",
    "instruct_type": None,
}


def _admin_headers() -> dict[str, str]:
    token = "test-admin-provenance-token"
    admin_sessions[token] = int(
        (datetime.now(timezone.utc) + timedelta(minutes=5)).timestamp()
    )
    return {"Authorization": f"Bearer {token}"}


def _payload(
    provider_id: int,
    *,
    model_id: str = "prov-model",
    prompt: float = 1.4e-7,
    completion: float = 2.8e-7,
    enabled: bool = True,
    pricing: dict[str, object] | None = None,
    **extra: object,
) -> dict[str, object]:
    body: dict[str, object] = {
        "id": model_id,
        "name": "Prov Model",
        "description": "d",
        "created": 0,
        "context_length": 128000,
        "architecture": _ARCHITECTURE,
        "pricing": pricing
        if pricing is not None
        else {
            "prompt": prompt,
            "completion": completion,
            "request": 0.0,
            "image": 0.0,
            "web_search": 0.0,
            "internal_reasoning": 0.0,
            "input_cache_read": 0.0,
            "input_cache_write": 0.0,
        },
        "per_request_limits": None,
        "top_provider": None,
        "upstream_provider_id": provider_id,
        "canonical_slug": None,
        "alias_ids": [],
        "enabled": enabled,
        "forwarded_model_id": model_id,
    }
    body.update(extra)
    return body


async def _make_provider(session: AsyncSession) -> int:
    provider = UpstreamProviderRow(
        provider_type="generic",
        base_url="https://prov-upstream.example/v1",
        api_key="test-key",
        provider_fee=1.0,
    )
    session.add(provider)
    await session.commit()
    await session.refresh(provider)
    await reinitialize_upstreams()
    assert provider.id is not None
    return provider.id


def _seed_row(
    provider_id: int,
    *,
    model_id: str,
    pricing: dict[str, object],
    pricing_source: str | None,
    enabled: bool = True,
) -> ModelRow:
    return ModelRow(
        id=model_id,
        name=model_id,
        description="d",
        created=0,
        context_length=131072,
        architecture=json.dumps(_ARCHITECTURE),
        pricing=json.dumps(pricing),
        upstream_provider_id=provider_id,
        enabled=enabled,
        forwarded_model_id=model_id,
        pricing_source=pricing_source,
    )


# ---------------------------------------------------------------------------
# what a write claims
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_hand_added_priced_model_is_operator_entered(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    provider_id = await _make_provider(integration_session)

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(provider_id, model_id="hand-made"),
    )

    assert resp.status_code == 200
    row = await integration_session.get(ModelRow, ("hand-made", provider_id))
    assert row is not None
    assert row.pricing_source == "manual"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_create_that_declares_a_source_keeps_it(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """A "save as fetched" carries the provenance the fetch resolved; adopting
    it is what stops an import being laundered into an operator's own price."""
    provider_id = await _make_provider(integration_session)

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(provider_id, model_id="as-fetched", pricing_source="litellm"),
    )

    assert resp.status_code == 200
    row = await integration_session.get(ModelRow, ("as-fetched", provider_id))
    assert row is not None
    assert row.pricing_source == "litellm"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_editing_a_price_makes_it_operator_entered(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    provider_id = await _make_provider(integration_session)
    headers = _admin_headers()
    await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=headers,
        json=_payload(provider_id, model_id="edit-me", pricing_source="litellm"),
    )

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=headers,
        json=_payload(
            provider_id, model_id="edit-me", prompt=9.9e-7, pricing_source="litellm"
        ),
    )

    assert resp.status_code == 200
    row = await integration_session.get(ModelRow, ("edit-me", provider_id))
    assert row is not None
    assert row.pricing_source == "manual"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_non_price_edit_leaves_the_resolved_source_alone(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """Renaming a model does not make its price operator-entered."""
    provider_id = await _make_provider(integration_session)
    headers = _admin_headers()
    await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=headers,
        json=_payload(provider_id, model_id="rename-me", pricing_source="openrouter"),
    )

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=headers,
        json=_payload(provider_id, model_id="rename-me", name="Renamed"),
    )

    assert resp.status_code == 200
    row = await integration_session.get(ModelRow, ("rename-me", provider_id))
    assert row is not None
    assert row.name == "Renamed"
    assert row.pricing_source == "openrouter"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_declared_source_replaces_the_one_the_row_claimed(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """A payload that names a source wins over what the row already recorded,
    even when that was an operator's own ``manual``.

    This is the price of letting "save as fetched" refresh provenance: the write
    cannot tell a refresh from a stale tag a client happened to send back, so a
    declared source is taken at face value. It costs nothing in money — the
    price is unchanged either way, and both tags read as "something vouches for
    this" — but it does mean a vouch is only as durable as the next write that
    names a source. An operator who wants the vouch back re-enters the price.
    """
    provider_id = await _make_provider(integration_session)
    row = _seed_row(
        provider_id,
        model_id="vouched-free",
        pricing={"prompt": 0.0, "completion": 0.0},
        pricing_source="manual",
    )
    integration_session.add(row)
    await integration_session.commit()

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(
            provider_id,
            model_id="vouched-free",
            prompt=0.0,
            completion=0.0,
            pricing_source="litellm",
        ),
    )

    assert resp.status_code == 200
    await integration_session.refresh(row)
    assert row.pricing_source == "litellm"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resaving_the_view_that_was_shown_is_not_an_edit(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """The view an operator is served backfills cache rates the stored price
    never had. Comparing a faithful re-save against the raw stored JSON would
    read those backfilled rates as an edit and relabel the row on every save."""
    provider_id = await _make_provider(integration_session)
    row = _seed_row(
        provider_id,
        model_id="deepseek-chat",
        pricing={"prompt": 2.8e-7, "completion": 4.2e-7},
        pricing_source="litellm",
    )
    integration_session.add(row)
    await integration_session.commit()

    view = _row_to_model(row, apply_provider_fee=False)
    assert view.pricing.input_cache_read > 0  # the backfill really happened

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(
            provider_id, model_id="deepseek-chat", pricing=view.pricing.dict()
        ),
    )

    assert resp.status_code == 200
    await integration_session.refresh(row)
    assert row.pricing_source == "litellm"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_omitting_a_backfilled_cache_rate_is_not_an_edit(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """A client that saves an unchanged price without the cache rates — they
    exist only via the read-time backfill, never in the stored JSON — has not
    changed the effective price: storing zero for them is re-backfilled on the
    next read. Comparing like for like is what keeps that from reading as an
    edit."""
    provider_id = await _make_provider(integration_session)
    row = _seed_row(
        provider_id,
        model_id="deepseek-chat",
        pricing={"prompt": 2.8e-7, "completion": 4.2e-7},
        pricing_source="litellm",
    )
    integration_session.add(row)
    await integration_session.commit()

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(
            provider_id,
            model_id="deepseek-chat",
            pricing={"prompt": 2.8e-7, "completion": 4.2e-7},
        ),
    )

    assert resp.status_code == 200
    await integration_session.refresh(row)
    assert row.pricing_source == "litellm"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dropping_a_priced_rate_is_an_edit(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """A write replaces the whole stored price, so a payload that omits a priced
    rate really does drop it to zero. A trusted tag must not survive a price the
    write silently changed — ``request`` has no backfilled twin to hide behind.
    """
    provider_id = await _make_provider(integration_session)
    row = _seed_row(
        provider_id,
        model_id="req-priced-xyz",
        pricing={"prompt": 1e-7, "completion": 2e-7, "request": 0.5},
        pricing_source="openrouter",
    )
    integration_session.add(row)
    await integration_session.commit()

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(
            provider_id,
            model_id="req-priced-xyz",
            pricing={"prompt": 1e-7, "completion": 2e-7},
        ),
    )

    assert resp.status_code == 200
    await integration_session.refresh(row)
    stored = _row_to_model(row, apply_provider_fee=False)
    assert stored.pricing.request == 0.0
    assert row.pricing_source == "manual"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_string_typed_price_edit_is_still_an_edit(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """Some clients emit rates as strings. The stored JSON is parsed as float on
    read, so the comparison has to read them the same way or an operator's edit
    keeps a stale trusted tag."""
    provider_id = await _make_provider(integration_session)
    headers = _admin_headers()
    await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=headers,
        json=_payload(provider_id, model_id="str-edit", pricing_source="litellm"),
    )

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=headers,
        json=_payload(
            provider_id,
            model_id="str-edit",
            pricing_source="litellm",
            pricing={"prompt": "9.9e-07", "completion": "2.8e-07"},
        ),
    )

    assert resp.status_code == 200
    row = await integration_session.get(ModelRow, ("str-edit", provider_id))
    assert row is not None
    assert row.pricing_source == "manual"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_batch_override_resolves_provenance_the_same_way(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    provider_id = await _make_provider(integration_session)
    headers = _admin_headers()

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/batch-override",
        headers=headers,
        json={
            "models": [
                _payload(provider_id, model_id="batch-a", pricing_source="openrouter")
            ]
        },
    )
    assert resp.status_code == 200
    row = await integration_session.get(ModelRow, ("batch-a", provider_id))
    assert row is not None and row.pricing_source == "openrouter"

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/batch-override",
        headers=headers,
        json={
            "models": [
                _payload(
                    provider_id,
                    model_id="batch-a",
                    prompt=5e-7,
                    pricing_source="openrouter",
                )
            ]
        },
    )

    assert resp.status_code == 200
    await integration_session.refresh(row)
    assert row.pricing_source == "manual"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_source_that_is_not_a_source_is_rejected(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """A junk tag would be persisted and then read back as nothing — provenance
    lost silently, and a zero price left without the source the write guard
    needs to judge it. Report the client bug instead."""
    provider_id = await _make_provider(integration_session)

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(provider_id, model_id="bad-src", pricing_source="lite-llm"),
    )

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# a row that cannot be read must still be repairable
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_model_with_an_unreadable_price_can_be_repaired(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """Deciding whether the price was edited means reading the price that is
    there. When that read fails, the row an operator most needs to fix would be
    the one row the API refuses a fix for — so a failed read answers "nothing to
    compare against" rather than aborting the write.
    """
    provider_id = await _make_provider(integration_session)
    integration_session.add(
        _seed_row(
            provider_id,
            model_id="repair-me",
            pricing={"prompt": "not-a-number", "completion": 2e-06},
            pricing_source="unresolved",
            enabled=False,
        )
    )
    await integration_session.commit()

    try:
        resp = await integration_client.post(
            f"/admin/api/upstream-providers/{provider_id}/models",
            headers=_admin_headers(),
            json=_payload(
                provider_id,
                model_id="repair-me",
                prompt=1e-06,
                completion=2e-06,
                enabled=True,
            ),
        )
        repair_status = resp.status_code
    except ValidationError:
        # The integration transport re-raises what production turns into a 500.
        # Normalise it so the regression fails on the API contract below rather
        # than looking like fixture breakage.
        repair_status = 500

    assert repair_status == 200
    integration_session.expire_all()
    row = await integration_session.get(ModelRow, ("repair-me", provider_id))
    assert row is not None
    assert json.loads(row.pricing)["prompt"] == pytest.approx(1e-06)
    assert row.pricing_source == "manual"
    assert row.enabled is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_one_unreadable_price_does_not_abort_a_whole_batch(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """The batch path snapshots the same view, so it needs the same recovery —
    and it matters more there: one poisoned row would block the repair of every
    model in the write."""
    provider_id = await _make_provider(integration_session)
    integration_session.add(
        _seed_row(
            provider_id,
            model_id="repair-batch",
            pricing={"prompt": None, "completion": 2e-06},
            pricing_source="unresolved",
            enabled=False,
        )
    )
    await integration_session.commit()

    try:
        resp = await integration_client.post(
            f"/admin/api/upstream-providers/{provider_id}/batch-override",
            headers=_admin_headers(),
            json={
                "models": [
                    _payload(
                        provider_id,
                        model_id="repair-batch",
                        prompt=1e-06,
                        completion=2e-06,
                        enabled=True,
                    ),
                    _payload(provider_id, model_id="healthy-sibling"),
                ]
            },
        )
        repair_status = resp.status_code
    except ValidationError:
        repair_status = 500

    assert repair_status == 200
    integration_session.expire_all()
    row = await integration_session.get(ModelRow, ("repair-batch", provider_id))
    assert row is not None
    assert json.loads(row.pricing)["prompt"] == pytest.approx(1e-06)
    assert row.pricing_source == "manual"
    assert row.enabled is True
    sibling = await integration_session.get(ModelRow, ("healthy-sibling", provider_id))
    assert sibling is not None


# ---------------------------------------------------------------------------
# the zero-price interlock, and where it does not run
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enabling_a_model_no_source_priced_is_refused(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """A model nothing could price is imported at zero and disabled. Enabling it
    without pricing it first would serve it and bill every request nothing."""
    provider_id = await _make_provider(integration_session)

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(
            provider_id,
            model_id="unpriced",
            prompt=0.0,
            completion=0.0,
            pricing_source="unresolved",
            enabled=True,
        ),
    )

    assert resp.status_code == 400
    assert "unpriced" in resp.json()["detail"]
    assert await integration_session.get(ModelRow, ("unpriced", provider_id)) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_creating_an_enabled_free_model_without_a_source_is_refused(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """A client that sends no provenance cannot tell a deliberate free import
    from an unpriced one, so a zero price arrives as ``unresolved`` and the same
    refusal applies."""
    provider_id = await _make_provider(integration_session)

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(
            provider_id, model_id="free-untagged", prompt=0.0, completion=0.0
        ),
    )

    assert resp.status_code == 400


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_model_no_source_priced_may_still_be_stored_disabled(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """The check is about serving, not about storing. An unpriced model still
    has to be importable so an operator can find it and price it."""
    provider_id = await _make_provider(integration_session)

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(
            provider_id,
            model_id="parked",
            prompt=0.0,
            completion=0.0,
            enabled=False,
        ),
    )

    assert resp.status_code == 200
    row = await integration_session.get(ModelRow, ("parked", provider_id))
    assert row is not None
    assert row.pricing_source == "unresolved"
    assert row.enabled is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pricing_an_unpriced_model_is_what_lets_it_be_enabled(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """The refusal names a way out, and this is it: give the model a price. That
    makes the price the operator's own, which satisfies the check."""
    provider_id = await _make_provider(integration_session)
    integration_session.add(
        _seed_row(
            provider_id,
            model_id="price-me",
            pricing={"prompt": 0.0, "completion": 0.0},
            pricing_source="unresolved",
            enabled=False,
        )
    )
    await integration_session.commit()

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(provider_id, model_id="price-me", enabled=True),
    )

    assert resp.status_code == 200
    integration_session.expire_all()
    row = await integration_session.get(ModelRow, ("price-me", provider_id))
    assert row is not None
    assert row.pricing_source == "manual"
    assert row.enabled is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_an_operator_may_vouch_for_a_model_that_is_genuinely_free(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """Free is a real price. An operator who declares the price their own is
    making a deliberate statement, and the check defers to it."""
    provider_id = await _make_provider(integration_session)

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(
            provider_id,
            model_id="really-free",
            prompt=0.0,
            completion=0.0,
            pricing_source="manual",
            enabled=True,
        ),
    )

    assert resp.status_code == 200
    row = await integration_session.get(ModelRow, ("really-free", provider_id))
    assert row is not None
    assert row.enabled is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_an_operator_zeroing_a_price_owns_that_choice(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """Editing a price down to zero is a price edit, so the price becomes the
    operator's own and the model stays enabled."""
    provider_id = await _make_provider(integration_session)
    integration_session.add(
        _seed_row(
            provider_id,
            model_id="zero-me",
            pricing={"prompt": 1e-7, "completion": 2e-7},
            pricing_source="openrouter",
        )
    )
    await integration_session.commit()

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(
            provider_id,
            model_id="zero-me",
            prompt=0.0,
            completion=0.0,
            enabled=True,
        ),
    )

    assert resp.status_code == 200
    integration_session.expire_all()
    row = await integration_session.get(ModelRow, ("zero-me", provider_id))
    assert row is not None
    assert row.pricing_source == "manual"
    assert row.enabled is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_price_only_a_request_rate_can_bill_does_not_count_as_priced(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """The reservation is derived from the token rates alone, so a model priced
    purely per request is advertised, reserves the floor and settles there. The
    check must not vouch for a collection the node cannot make."""
    provider_id = await _make_provider(integration_session)

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/models",
        headers=_admin_headers(),
        json=_payload(
            provider_id,
            model_id="request-only",
            pricing={"prompt": 0.0, "completion": 0.0, "request": 0.5},
            pricing_source="unresolved",
            enabled=True,
        ),
    )

    assert resp.status_code == 400


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_batch_naming_unpriced_models_is_refused_whole(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """The batch write is one write. Refusing it whole — naming every offending
    model — is what keeps an operator from having to guess which half landed."""
    provider_id = await _make_provider(integration_session)

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/batch-override",
        headers=_admin_headers(),
        json={
            "models": [
                _payload(provider_id, model_id="batch-priced"),
                _payload(
                    provider_id,
                    model_id="batch-unpriced-a",
                    prompt=0.0,
                    completion=0.0,
                    pricing_source="unresolved",
                ),
                _payload(
                    provider_id,
                    model_id="batch-unpriced-b",
                    prompt=0.0,
                    completion=0.0,
                    pricing_source="unresolved",
                ),
            ]
        },
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "batch-unpriced-a" in detail
    assert "batch-unpriced-b" in detail
    integration_session.expire_all()
    for model_id in ("batch-priced", "batch-unpriced-a", "batch-unpriced-b"):
        assert await integration_session.get(ModelRow, (model_id, provider_id)) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_refused_batch_leaves_a_row_it_already_touched_alone(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """Refusing the batch whole has to mean the rows already written in the loop
    are rolled back, not just that the offending one was skipped.

    The loop mutates rows as it walks the payload, so a model reached *before*
    the offending one has had its price and its provenance overwritten in the
    session by the time the refusal is raised. Only never reaching the commit
    keeps that from landing — which is why the write runs in a session of its
    own rather than one handed in by the caller.
    """
    provider_id = await _make_provider(integration_session)
    integration_session.add(
        _seed_row(
            provider_id,
            model_id="already-priced",
            pricing={"prompt": 1e-06, "completion": 2e-06},
            pricing_source="litellm",
            enabled=True,
        )
    )
    await integration_session.commit()

    resp = await integration_client.post(
        f"/admin/api/upstream-providers/{provider_id}/batch-override",
        headers=_admin_headers(),
        json={
            "models": [
                _payload(
                    provider_id,
                    model_id="already-priced",
                    prompt=9.9e-06,
                    completion=9.9e-06,
                ),
                _payload(
                    provider_id,
                    model_id="unpriced",
                    prompt=0.0,
                    completion=0.0,
                    pricing_source="unresolved",
                ),
            ]
        },
    )

    assert resp.status_code == 400
    integration_session.expire_all()
    row = await integration_session.get(ModelRow, ("already-priced", provider_id))
    assert row is not None
    assert json.loads(row.pricing)["prompt"] == pytest.approx(1e-06)
    assert row.pricing_source == "litellm"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_stored_unpriced_row_that_is_enabled_is_still_served(
    integration_client: AsyncClient, integration_session: AsyncSession
) -> None:
    """The check runs at the write edge and nowhere else.

    Rows that were already enabled and priced at nothing keep being served until
    an operator touches them. That is the accepted cost of keeping the question
    "is this a price?" at the one place it can be answered without asking the
    much larger question of whether a request in flight can be collected on.
    """
    provider_id = await _make_provider(integration_session)
    integration_session.add(
        _seed_row(
            provider_id,
            model_id="legacy-free",
            pricing={"prompt": 0.0, "completion": 0.0},
            pricing_source="unresolved",
            enabled=True,
        )
    )
    await integration_session.commit()

    from routstr.payment.models import list_models

    served = {m.id for m in await list_models(integration_session, provider_id)}

    assert served == {"legacy-free"}
