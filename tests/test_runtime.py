import os

from aeon_v1.runtime import process_alive


def test_process_alive_detects_current_process():
    assert process_alive(os.getpid()) is True


def test_process_alive_rejects_invalid_pid():
    assert process_alive(0) is False
    assert process_alive(-1) is False
