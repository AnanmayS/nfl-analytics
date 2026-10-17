import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.game import linear_contributions


def test_contributions_sum_to_decision_function():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)) * [1, 5, 0.1, 2] + [0, 3, 0, -1]
    y = (X[:, 0] + 0.2 * X[:, 1] + rng.normal(size=200) > 0.5).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1)).fit(X, y)
    contrib, intercept = linear_contributions(model, X[:10])
    np.testing.assert_allclose(contrib.sum(1) + intercept, model.decision_function(X[:10]))
