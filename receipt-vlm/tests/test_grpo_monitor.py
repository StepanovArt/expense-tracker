from receipt_vlm.train.grpo import HackingMonitor


def test_flags_null_rate_rising_with_reward() -> None:
    m = HackingMonitor(window=3, min_rise=0.05)
    flags = [
        m.update({"reward": 0.5 + 0.02 * i, "null_rate/all": 0.1 + 0.03 * i}) for i in range(6)
    ]
    assert flags[:5] == [None] * 5 and flags[5] is not None
    assert flags[5]["null_rate_rise"] > 0.05 and flags[5]["reward_rise"] > 0


def test_no_flag_when_nulls_fall_or_reward_falls() -> None:
    m = HackingMonitor(window=3, min_rise=0.05)
    assert all(
        m.update({"reward": 0.5 + 0.02 * i, "null_rate/all": 0.5 - 0.03 * i}) is None
        for i in range(8)
    )
    m = HackingMonitor(window=3, min_rise=0.05)
    assert all(
        m.update({"reward": 0.9 - 0.02 * i, "null_rate/all": 0.1 + 0.05 * i}) is None
        for i in range(8)
    )
    assert HackingMonitor().update({"loss": 1.0}) is None  # logs without reward are ignored
