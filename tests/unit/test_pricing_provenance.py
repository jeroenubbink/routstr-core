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


# ---------------------------------------------------------------------------
# a backfilled cache rate makes the price mixed-source
# ---------------------------------------------------------------------------


def _fee_rebuild(model: Model) -> Model:
    return GenericUpstreamProvider(base_url="http://x")._apply_provider_fee_to_model(
        model
    )


def test_a_backfilled_cache_rate_stops_a_price_claiming_native() -> None:
    """The tag speaks for the whole price, so it can only honestly name the
    least-trusted source that contributed a rate to it. A price the provider
    supplied, wearing a cache rate the bundled cost map supplied, is no longer
    wholly the provider's."""
    model = _model_with_source(PricingSource.NATIVE, model_id="gpt-4o")
    assert model.pricing.input_cache_read == 0.0

    rebuilt = _fee_rebuild(model)

    assert rebuilt.pricing.input_cache_read > 0.0
    assert rebuilt.pricing_source is PricingSource.LITELLM


def test_a_backfill_never_promotes_a_less_trusted_price() -> None:
    """A backfill is not evidence in a price's favour: an openrouter price is
    already below litellm, so mixing one in must change nothing."""
    model = _model_with_source(PricingSource.OPENROUTER, model_id="gpt-4o")

    rebuilt = _fee_rebuild(model)

    assert rebuilt.pricing.input_cache_read > 0.0
    assert rebuilt.pricing_source is PricingSource.OPENROUTER


def test_a_backfill_does_not_revoke_an_operator_vouch() -> None:
    """``manual`` is the operator's own statement about the price, and it is
    what lets a deliberately free model be enabled at all. A cache lookup must
    not silently strip that vouch."""
    model = _model_with_source(PricingSource.MANUAL, model_id="gpt-4o")

    rebuilt = _fee_rebuild(model)

    assert rebuilt.pricing.input_cache_read > 0.0
    assert rebuilt.pricing_source is PricingSource.MANUAL


def test_a_price_the_cost_map_cannot_touch_keeps_its_own_claim() -> None:
    """Only an actual contribution downgrades the tag."""
    model = _model_with_source(PricingSource.NATIVE, model_id="m1")

    rebuilt = _fee_rebuild(model)

    assert rebuilt.pricing.input_cache_read == 0.0
    assert rebuilt.pricing_source is PricingSource.NATIVE


def test_reading_a_row_back_owes_the_same_correction() -> None:
    """The database read path backfills the same rates from the same source, so
    a stored ``native`` row read back with a borrowed cache rate is mixed-source
    exactly as the fetch path's is."""
    model = _row_to_model(
        _row("gpt-4o", {"prompt": 5e-06, "completion": 1.5e-05}, "native")
    )

    assert model.pricing.input_cache_read > 0.0
    assert model.pricing_source is PricingSource.LITELLM


# ---------------------------------------------------------------------------
# ppqai — its own published price is native, and only when it is a whole price
# ---------------------------------------------------------------------------


def _or_entry(model_id: str, **pricing: float) -> dict[str, Any]:
    rates: dict[str, float] = {"prompt": 0.000001, "completion": 0.000002}
    rates.update(pricing)
    return {
        "id": model_id,
        "name": model_id,
        "created": 0,
        "description": "d",
        "context_length": 8192,
        "architecture": _ARCHITECTURE,
        "pricing": rates,
        "pricing_source": "openrouter",
    }


def _ppq_entry(model_id: str, **api: float) -> dict[str, Any]:
    return {
        "id": model_id,
        "name": model_id,
        "created_at": 0,
        "context_length": 8192,
        "pricing": {"api": api} if api else {},
    }


async def _fetch_ppq(entries: list[dict[str, Any]], or_feed: list[dict]) -> list[Model]:
    from routstr.upstream.ppqai import PPQAIUpstreamProvider

    provider = PPQAIUpstreamProvider(api_key="k")
    with patch(
        "routstr.upstream.ppqai.httpx.AsyncClient",
        lambda *a, **k: _FakeAsyncClient({"data": entries}),
    ):
        with patch(
            "routstr.upstream.ppqai.async_fetch_openrouter_models",
            AsyncMock(return_value=or_feed),
        ):
            return await provider.fetch_models()


@pytest.mark.asyncio
async def test_a_ppq_published_price_is_native_and_an_absent_one_is_not() -> None:
    models = await _fetch_ppq(
        [
            _ppq_entry("ppq-priced", input_per_1M=1.0, output_per_1M=2.0),
            _ppq_entry("ppq-free"),
        ],
        [],
    )

    assert _model_by_id(models, "ppq-priced").pricing_source is PricingSource.NATIVE
    unpriced = _model_by_id(models, "ppq-free")
    assert unpriced.pricing_source is PricingSource.UNRESOLVED
    assert unpriced.enabled is False


@pytest.mark.asyncio
async def test_a_half_priced_ppq_model_is_not_a_native_price() -> None:
    """Billing the unpriced side at nothing is not a price PPQ published, so the
    model fails closed rather than being served as confidently native."""
    models = await _fetch_ppq([_ppq_entry("ppq-input-only", input_per_1M=1.0)], [])

    model = _model_by_id(models, "ppq-input-only")
    assert model.pricing_source is PricingSource.UNRESOLVED
    assert model.enabled is False


@pytest.mark.asyncio
async def test_a_negative_ppq_rate_is_not_a_price() -> None:
    """A negative rate is malformed upstream data, and it is truthy: published
    as a price it would credit the caller per token."""
    models = await _fetch_ppq(
        [_ppq_entry("ppq-negative", input_per_1M=-1.0, output_per_1M=-2.0)], []
    )

    model = _model_by_id(models, "ppq-negative")
    assert model.pricing_source is PricingSource.UNRESOLVED
    assert model.enabled is False
    assert model.pricing.prompt >= 0
    assert model.pricing.completion >= 0


@pytest.mark.asyncio
async def test_two_ppq_ids_matching_one_feed_entry_keep_their_own_prices() -> None:
    """Two PPQ ids can tail-match the same feed entry. Overlaying onto the entry
    in place would leave both models pointing at one object, so the last writer's
    price and provenance would silently become the other's too."""
    models = await _fetch_ppq(
        [
            _ppq_entry("gpt-4o", input_per_1M=5.0, output_per_1M=15.0),
            _ppq_entry("openai/gpt-4o", input_per_1M=3.0, output_per_1M=10.0),
        ],
        [_or_entry("openai/gpt-4o")],
    )

    assert len(models) == 2
    assert models[0] is not models[1]
    assert {round(m.pricing.prompt * 1_000_000, 6) for m in models} == {5.0, 3.0}


@pytest.mark.asyncio
async def test_a_ppq_price_for_one_side_only_leaves_the_feed_as_the_source() -> None:
    """The other side is still the feed's, so the whole-price tag cannot claim
    the model is priced by the provider."""
    models = await _fetch_ppq(
        [_ppq_entry("gpt-4o", input_per_1M=5.0)], [_or_entry("gpt-4o")]
    )

    assert _model_by_id(models, "gpt-4o").pricing_source is PricingSource.OPENROUTER


@pytest.mark.asyncio
async def test_a_ppq_zero_does_not_overwrite_the_feed_price() -> None:
    """PPQ reports an unpriced side as 0. Overlaying that would bill those
    tokens at nothing; the feed's price for that side must stand."""
    models = await _fetch_ppq(
        [_ppq_entry("gpt-4o", input_per_1M=0, output_per_1M=15.0)],
        [_or_entry("gpt-4o")],
    )

    model = _model_by_id(models, "gpt-4o")
    assert model.pricing.prompt == pytest.approx(0.000001)
    assert model.pricing.completion == pytest.approx(15.0 / 1_000_000)
    assert model.pricing_source is PricingSource.OPENROUTER


@pytest.mark.asyncio
async def test_a_negative_ppq_rate_does_not_overwrite_the_feed_price() -> None:
    models = await _fetch_ppq(
        [_ppq_entry("gpt-4o", input_per_1M=-5.0, output_per_1M=-15.0)],
        [_or_entry("gpt-4o")],
    )

    model = _model_by_id(models, "gpt-4o")
    assert model.pricing.prompt == pytest.approx(0.000001)
    assert model.pricing.completion == pytest.approx(0.000002)
    assert model.pricing_source is PricingSource.OPENROUTER


@pytest.mark.asyncio
async def test_a_native_ppq_price_carries_no_rates_ppq_never_published() -> None:
    """PPQ publishes token rates and nothing else. Overlaying those two onto a
    matched feed entry left its request, image, search, reasoning and cache
    rates in place — so the model was billed auxiliary fees PPQ does not charge
    while the tag claimed every rate was the provider's own."""
    models = await _fetch_ppq(
        [_ppq_entry("gpt-4o", input_per_1M=5.0, output_per_1M=15.0)],
        [
            _or_entry(
                "gpt-4o",
                request=0.25,
                image=0.5,
                web_search=0.75,
                internal_reasoning=0.0000003,
                input_cache_read=0.0000001,
                input_cache_write=0.0000002,
            )
        ],
    )

    model = _model_by_id(models, "gpt-4o")
    assert model.pricing_source is PricingSource.NATIVE
    assert model.pricing.prompt == pytest.approx(5.0 / 1_000_000)
    assert model.pricing.completion == pytest.approx(15.0 / 1_000_000)
    assert model.pricing.request == 0.0
    assert model.pricing.image == 0.0
    assert model.pricing.web_search == 0.0
    assert model.pricing.internal_reasoning == 0.0
    assert model.pricing.input_cache_read == 0.0
    assert model.pricing.input_cache_write == 0.0
