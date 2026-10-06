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


def test_get_empty():
    assert not mime.get('')


def test_for_path_common():
    assert mime.for_path('image.jpg') == 'image/jpeg'


def test_for_path_bin():
    assert mime.for_path('') == mime.BIN


def test_for_path_guessed():
    assert mime.for_path('model.obj') == 'model/obj'


def test_extension_for():
    assert mime.extension_for('image/png') == 'png'


def test_extension_for_empty():
    assert mime.extension_for('') == 'bin'


def test_extension_for_guessed():
    assert mime.extension_for('model/obj') == 'obj'


def test_get_ooxml():
    assert mime.get('application/vnd.openxmlformats-officedocument.wordprocessingml.document') == mime.DOCX
    assert mime.get('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') == mime.XLSX
    assert mime.get('application/vnd.openxmlformats-officedocument.presentationml.presentation') == mime.PPTX


def test_get_ooxml_extension():
    assert mime.get('docx') == mime.DOCX
    assert mime.get('xlsx') == mime.XLSX
    assert mime.get('pptx') == mime.PPTX


def test_extension_for_ooxml():
    assert mime.extension_for(mime.DOCX) == 'docx'
    assert mime.extension_for(mime.XLSX) == 'xlsx'
    assert mime.extension_for(mime.PPTX) == 'pptx'
