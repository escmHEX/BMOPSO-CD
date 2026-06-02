from __future__ import annotations

from binary_mopso_cd.monitor import ObservationalMonitor


def test_disabled_monitor_has_no_overhead_dependency():
    result = ObservationalMonitor(enabled=False).observe(1, [])
    assert result.metrics == {"generation": 1}
    assert result.overhead_seconds == 0.0

