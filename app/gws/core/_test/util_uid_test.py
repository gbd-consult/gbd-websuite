"""Tests for uid helpers."""

import gws


def test_split_uid():
    assert gws.u.split_uid('a::b') == ('a', 'b')


def test_split_uid_at_first_delimiter():
    assert gws.u.split_uid('a::b::c') == ('a', 'b::c')


def test_split_uid_without_delimiter():
    assert gws.u.split_uid('a') == ('', 'a')
