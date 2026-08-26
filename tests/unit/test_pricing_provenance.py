"""Tests for pricing provenance — ``pricing_source`` on ``Model``.

Provenance makes a price's origin a first-class, queryable fact: ``native`` is
the provider's own (trustworthy) price, ``litellm``/``openrouter`` are curated
or resale estimates, ``manual`` is operator-entered, and ``unresolved`` marks a
model no source could price (imported disabled). These tests drive the tag
through the public provider ``fetch_models`` API and assert it survives the fee
and sats carrier rebuilds a catalog refresh performs.
"""

from __future__ import annotations

from routstr.payment.models import (
    Architecture,
    Model,
    Pricing,
    PricingSource,
    TopProvider,
    _update_model_sats_pricing,
    pricing_metadata,
)

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
