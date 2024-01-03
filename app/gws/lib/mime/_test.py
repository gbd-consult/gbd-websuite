"""Tests for the mime module."""

import gws
import gws.test.util as u
import gws.lib.mime as mime


def test_get_alias():
    assert mime.get('application/html') == 'text/html'


def test_get_semicolon():
    assert mime.get('text/html; charset=UTF-8') == 'text/html'


def test_get_path():
    assert mime.get('image.jpg') == 'image/jpeg'


def test_for_path():
    assert mime.for_path('image.jpg') == 'image/jpeg'


def test_extension_for():
    assert mime.extension_for('image/png') == 'png'
