import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from constrained_evasion.attacks import Moeva2
from constrained_evasion.objective_calculator.objective_calculator import (
    ObjectiveCalculator,
)
from tests.attacks.moeva.url_constraints import UrlConstraints


def test_run():
    constraints = UrlConstraints()
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")[
        :10
    ]
    y_clean = np.load("tests/resources/url/baseline_y_test_candidates.npy")[
        :10
    ]
    model = joblib.load("./tests/resources/url/baseline.model")
    preprocessing_pipeline = joblib.load(
        "./tests/resources/url/baseline_scaler.joblib"
    )
    model_pipeline = Pipeline(
        steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    attack = Moeva2(
        model_pipeline,
        constraints,
        2,
        preprocessing_pipeline.transform,
        save_history="full",
        seed=42,
        n_jobs=10,
    )
    out = attack.generate(x_clean, y_clean)
    assert len(out) == 2


def test_objective_calculation():
    constraints = UrlConstraints()
    x_clean = np.load("tests/resources/url/baseline_X_test_candidates.npy")[
        :10
    ]
    y_clean = np.load("tests/resources/url/baseline_y_test_candidates.npy")[
        :10
    ]
    model = joblib.load("./tests/resources/url/baseline.model")
    preprocessing_pipeline = joblib.load(
        "./tests/resources/url/baseline_scaler.joblib"
    )
    model_pipeline = Pipeline(
        steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
    )

    x_adv = np.repeat(x_clean[:, np.newaxis, :], 5, axis=1)
    objective_calculator = ObjectiveCalculator(
        model_pipeline,
        constraints,
        thresholds={"misclassification": 0.5, "distance": 0.2},
        norm=2,
        fun_distance_preprocess=preprocessing_pipeline.transform,
    )
    success_rate = objective_calculator.success_rate_many(
        x_clean, y_clean, x_adv
    )
    for i in range(7):
        assert 0 <= success_rate[i] and success_rate[i] <= 1.0

    assert success_rate[0] == 1.0
    assert success_rate[2] == 1.0
    assert success_rate[3] == success_rate[1]
    assert success_rate[4] == 1.0
    assert success_rate[5] == success_rate[1]
    assert success_rate[6] == success_rate[1]
