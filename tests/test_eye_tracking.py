from driver_monitor.calibration.baseline import Baseline


def test_baseline_tracks_a_mean():
    baseline = Baseline(sample_count=2)
    baseline.add(2.0)
    baseline.add(4.0)
    assert baseline.complete
    assert baseline.mean == 3.0
