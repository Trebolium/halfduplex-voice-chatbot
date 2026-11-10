"""memwatch: limit applies without error and log_mem prints MLX stats."""
from voicebot import memwatch


def test_apply_limit_and_log(capsys):
    memwatch.apply_limit()
    memwatch.log_mem("unit-test")
    out = capsys.readouterr().out
    assert "[mem] after unit-test" in out
