from driver_monitor.calibration.baseline import Baseline


def test_baseline_does_not_collect_after_completion():
    baseline = Baseline(sample_count=1)
    baseline.add(1.0)
    baseline.add(99.0)
    assert baseline.samples == [1.0]
