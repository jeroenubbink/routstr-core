"""Tests for pricing provenance — ``pricing_source`` on ``Model``.

Provenance makes a price's origin a first-class, queryable fact: ``native`` is
the provider's own (trustworthy) price, ``litellm``/``openrouter`` are curated
or resale estimates, ``manual`` is operator-entered, and ``unresolved`` marks a
model no source could price (imported disabled). These tests drive the tag
through the public provider ``fetch_models`` API and assert it survives the fee
and sats carrier rebuilds a catalog refresh performs.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from routstr.core.db import ModelRow
from routstr.payment.models import (
    Architecture,
    Model,
    Pricing,
    PricingSource,
    TopProvider,
    _row_to_model,
    _update_model_sats_pricing,
    pricing_metadata,
)
from routstr.upstream.generic import GenericUpstreamProvider

_ARCHITECTURE = {
    "modality": "text->text",
    "input_modalities": ["text"],
    "output_modalities": ["text"],
    "tokenizer": "unknown",
    "instruct_type": None,
}


def _model_with_source(
    source: PricingSource, model_id: str = "m1", **pricing: float
) -> Model:
    return Model(
        id=model_id,
        name="M1",
        created=0,
        description="d",
        context_length=4096,
        architecture=Architecture.parse_obj(_ARCHITECTURE),
        pricing=Pricing(prompt=1e-06, completion=2e-06, **pricing),
        top_provider=TopProvider(context_length=4096, max_completion_tokens=2048),
        **pricing_metadata(source),
    )


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(
        self, url: str, headers: dict[str, str] | None = None
    ) -> _FakeResponse:
        return _FakeResponse(self._payload)


def _model_by_id(models: list[Any], model_id: str) -> Any:
    return next(m for m in models if m.id == model_id)


async def _fetch_generic(payload: dict[str, Any], or_feed: list[dict]) -> list[Model]:
    """Drive ``GenericUpstreamProvider.fetch_models`` over a canned ``/models``
    body and OpenRouter feed. litellm's real bundled cost map is left unmocked —
    the rates it ships for DeepSeek are the assertion's ground truth."""
    with patch(
        "routstr.upstream.generic.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(payload),
    ):
        with patch(
            "routstr.payment.models.async_fetch_openrouter_models",
            AsyncMock(return_value=or_feed),
        ):
            return await GenericUpstreamProvider(base_url="http://x").fetch_models()


def _row(model_id: str, pricing: dict[str, object], source: str | None) -> ModelRow:
    return ModelRow(
        id=model_id,
        name=model_id,
        created=0,
        description="",
        context_length=128000,
        architecture=json.dumps(_ARCHITECTURE),
        pricing=json.dumps(pricing),
        enabled=True,
        upstream_provider_id=1,
        pricing_source=source,
    )


# ---------------------------------------------------------------------------
# the tag survives the carrier rebuilds
# ---------------------------------------------------------------------------


def test_sats_pricing_rebuild_preserves_provenance() -> None:
    """``_update_model_sats_pricing`` rebuilds ``Model`` field by field, so it
    silently drops any field it was not taught about."""
    model = _model_with_source(PricingSource.LITELLM)

    rebuilt = _update_model_sats_pricing(model, sats_to_usd=0.0005)

    assert rebuilt.sats_pricing is not None
    assert rebuilt.pricing_source == PricingSource.LITELLM


# ---------------------------------------------------------------------------
# reading the tag back off a stored row
# ---------------------------------------------------------------------------


def test_stored_source_is_read_back_as_the_enum() -> None:
    model = _row_to_model(
        _row("m1", {"prompt": 1e-06, "completion": 2e-06}, "openrouter")
    )

    assert model.pricing_source is PricingSource.OPENROUTER


def test_unknown_stored_source_reads_as_unrecorded_instead_of_raising() -> None:
    """The read path runs on every catalog build. A value that is not a source —
    a foreign writer, a typo, a downgrade from a version that knew more — must
    not raise, or one row would blank the whole served catalog."""
    model = _row_to_model(
        _row("m1", {"prompt": 1e-06, "completion": 2e-06}, "wishful-thinking")
    )

    assert model.pricing_source is None


def test_a_row_written_before_provenance_reads_as_unrecorded() -> None:
    model = _row_to_model(_row("m1", {"prompt": 1e-06, "completion": 2e-06}, None))

    assert model.pricing_source is None


# ---------------------------------------------------------------------------
# the resolving path tags every source it actually used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_provider_priced_model_is_tagged_native() -> None:
    payload = {
        "data": [
            {
                "id": "venice-llama",
                "owned_by": "venice",
                "model_spec": {
                    "pricing": {"input": {"usd": 0.5}, "output": {"usd": 1.5}},
                },
            }
        ]
    }

    models = await _fetch_generic(payload, [])

    assert _model_by_id(models, "venice-llama").pricing_source is PricingSource.NATIVE


@pytest.mark.asyncio
async def test_a_model_priced_from_the_bundled_cost_map_is_tagged_litellm() -> None:
    payload = {"data": [{"id": "deepseek-chat", "owned_by": "deepseek"}]}

    models = await _fetch_generic(payload, [])

    assert _model_by_id(models, "deepseek-chat").pricing_source is PricingSource.LITELLM


@pytest.mark.asyncio
async def test_a_model_priced_from_the_feed_is_tagged_openrouter() -> None:
    payload = {"data": [{"id": "exotic/model-9000", "owned_by": "exotic"}]}
    or_feed = [
        {
            "id": "exotic/model-9000",
            "context_length": 65536,
            "pricing": {"prompt": "0.000005", "completion": "0.000010"},
        }
    ]

    models = await _fetch_generic(payload, or_feed)

    model = _model_by_id(models, "exotic/model-9000")
    assert model.pricing_source is PricingSource.OPENROUTER


@pytest.mark.asyncio
async def test_a_model_nothing_could_price_is_tagged_unresolved() -> None:
    """The fail-closed import already leaves such a model disabled; the tag is
    what lets an operator find it and tell it apart from a genuinely free one."""
    payload = {"data": [{"id": "nobody-has-priced-this-xyz", "owned_by": "mystery"}]}

    models = await _fetch_generic(payload, [])

    model = _model_by_id(models, "nobody-has-priced-this-xyz")
    assert model.enabled is False
    assert model.pricing_source is PricingSource.UNRESOLVED


@pytest.mark.asyncio
async def test_the_openrouter_feed_tags_every_entry_it_returns() -> None:
    """Tagging at the feed means the ``Model(**entry)`` spreads in the OpenRouter
    -fed providers (openai, xai, anthropic, perplexity, ...) inherit provenance
    with no per-provider code."""
    from routstr.payment import models as models_mod

    or_payload = {
        "data": [
            {
                "id": "openai/gpt-4o",
                "name": "GPT-4o",
                "pricing": {"prompt": "0.000005", "completion": "0.000015"},
            }
        ]
    }

    class _Client:
        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def get(self, url: str, timeout: int = 30) -> _FakeResponse:
            payload = or_payload if "embeddings" not in url else {"data": []}
            return _FakeResponse(payload)

    with patch.object(models_mod.httpx, "AsyncClient", lambda *a, **k: _Client()):
        feed = await models_mod.async_fetch_openrouter_models()

    assert [entry["pricing_source"] for entry in feed] == [PricingSource.OPENROUTER]


def test_the_provider_fee_rebuild_preserves_provenance() -> None:
    """Applying the provider fee rebuilds ``Model`` twice, field by field."""
    model = _model_with_source(PricingSource.NATIVE)

    rebuilt = GenericUpstreamProvider(base_url="http://x")._apply_provider_fee_to_model(
        model
    )

    assert rebuilt.pricing_source is PricingSource.NATIVE
