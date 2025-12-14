import pytest
import numpy as np
from scipy.stats import ks_2samp
from model_pipeline import load_data

DRIFT_P_THRESHOLD = 0.05


def test_data_drift():
    """
    Detects data drift between training and test datasets
    using Kolmogorov-Smirnov test.
    """
    X_train, X_test, _, _, _ = load_data()

    drift_detected = False

    for feature_idx in range(X_train.shape[1]):
        stat, p_value = ks_2samp(
            X_train[:, feature_idx],
            X_test[:, feature_idx]
        )

        if p_value < DRIFT_P_THRESHOLD:
            drift_detected = True
            break

    assert not drift_detected, "Data drift detected between train and test distributions"
