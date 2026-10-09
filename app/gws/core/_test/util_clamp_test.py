"""Tests for clamp."""

import gws


def test_clamp_inside():
    assert gws.u.clamp(5, 1, 10) == 5


def test_clamp_below():
    assert gws.u.clamp(0, 1, 10) == 1


def test_clamp_above():
    assert gws.u.clamp(11, 1, 10) == 10
