import numpy as np
import pytest
from condition.energy import fit, load, windows
from condition.vibration import experiment, simulate


def test_alignment_and_temporal_boundary():
    a = np.arange(240, dtype=np.float32).reshape(48, 5)
    x, y = windows(a, 20, 30, 12)
    np.testing.assert_array_equal(x[0], a[8:20])
    np.testing.assert_array_equal(y, a[20:30, 0])
    with pytest.raises(ValueError):
        windows(a, 20, 20, 12)


def test_untrusted_archive(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="digest"):
        load(p)


def test_both_forecast_models():
    r = fit(np.random.default_rng(5).normal(size=(200, 5)).astype("float32"), epochs=1)
    assert r["test_targets"] == 30
    for metrics in r["results"].values():
        assert metrics["mae_Wh"] >= 0 and 0 <= metrics["empirical_interval_coverage"] <= 1


def test_asset_experiment():
    features, spec, labels, assets = simulate(2)
    assert len(features) == 48 and np.isfinite(spec).all()
    assert set(assets) == {0, 1} and set(labels) == {0, 1, 2}
    results = experiment()
    for model in results["results"].values():
        assert np.sum(model["confusion"]) == 144 and 0 <= model["macro_f1"] <= 1


def test_thermal_and_parts_contract():
    from condition.thermography import experiment as thermal
    from condition.parts import measure

    t = thermal()
    assert t["n"] == 128 and t["held_out_assets"] == 8 and 0 <= t["cnn_f1"] <= 1
    p = measure()
    assert p["test_orders"] == 320 and len(p["recall_at_5"]) == 3
    assert all(0 <= r <= 1 for r in p["recall_at_5"].values())
