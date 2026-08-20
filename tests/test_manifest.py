from src.data.manifest import equal_spaced_starts


def test_equal_spaced_starts_include_bounds():
    starts = equal_spaced_starts(10000, 2048, 10)
    assert len(starts) == 10
    assert starts[0] == 0
    assert starts[-1] == 7952
    assert starts == sorted(set(starts))


def test_short_signal_has_no_windows():
    assert equal_spaced_starts(2047, 2048, 10) == []
