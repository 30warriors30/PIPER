from evaluation.calibration import calibrate_threshold


def test_calibration_uses_higher_quantile() -> None:
    assert calibrate_threshold([0.0, 1.0, 2.0, 3.0], 0.25) == 3.0
