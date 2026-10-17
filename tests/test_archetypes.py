import numpy as np
import polars as pl
import pytest

from src.archetypes.naming import describe, name_clusters
from src.archetypes.pipeline import standardize

FEATS = ["adot", "target_share", "avg_separation"]


def test_describe_orders_terms_by_word_position():
    # target_share is most extreme, but adjectives print before the volume noun
    terms = describe(np.array([-0.8, 1.2, 0.1]), FEATS)
    assert [t for t, _ in terms] == ["Alpha", "Underneath"]
    assert name_clusters(np.array([[-0.8, 1.2, 0.1]]), np.array([0]), FEATS) == {
        0: "Underneath Alpha"
    }


def test_weak_centroid_is_balanced():
    assert describe(np.array([0.1, -0.2, 0.0]), FEATS) == [("Balanced", 0)]


def test_names_unique_and_noise_ignored():
    Z = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [0.0, 0.0, 0.0]])
    names = name_clusters(Z, np.array([0, 1, -1]), FEATS)
    assert set(names) == {0, 1}
    assert len(set(names.values())) == 2


def test_standardize_uses_fit_statistics_only():
    fit = pl.DataFrame({"a": [1.0, 2.0, 3.0, None]})
    apply = pl.DataFrame({"a": [100.0, None]})
    Z, Zp = standardize(fit, apply, ["a"])
    # null in fit imputed with fit median (2.0); scaler mean = 2.0
    assert Z.mean() == pytest.approx(0.0)
    # projection nulls use the fit median too, and outliers don't shift the scale
    assert Zp[1, 0] == pytest.approx(0.0)
    assert Zp[0, 0] > 50
