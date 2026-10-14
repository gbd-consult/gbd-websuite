"""Tests for the mime module."""

import gws
import gws.test.util as u
import gws.lib.mime as mime


def test_get_alias():
    assert mime.get('application/html') == 'text/html'


def test_get_semicolon():
    assert mime.get('text/html; charset=UTF-8') == 'text/html; charset=UTF-8'


def test_get_alias_with_params():
    assert mime.get('image/jpg; quality=low') == 'image/jpeg; quality=low'


def test_get_known_with_params():
    assert mime.get('image/png;mode=8bit') == mime.PNG8
    assert mime.get('text/xml;subtype=foo') == 'text/xml; subtype=foo'


def test_get_png_aliases():
    assert mime.get('image/png8') == mime.PNG8
    assert mime.get('image/png32') == mime.PNG24
    assert mime.get('image/png; mode=24bit') == 'image/png; mode=24bit'


def test_get_unknown_with_params():
    assert mime.get('foo/bar; x=1') == ''


def test_get_base():
    assert mime.get_base('image/png8') == mime.PNG
    assert mime.get_base('application/vnd.ogc.gml') == mime.GML
    assert mime.get_base('text/html; charset=UTF-8') == mime.HTML
    assert mime.get_base('') == ''


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


def test_get_any_spelling():
    assert mime.get('Image/PNG;Mode="8bit"') == mime.PNG8
    assert mime.get(' image/png ; mode=8bit ') == mime.PNG8
    assert mime.get('text/xml;subtype=foo;') == 'text/xml; subtype=foo'
    assert mime.get('IMAGE/PNG8') == mime.PNG8


def test_get_gml3_any_spelling():
    assert mime.get('application/gml+xml;version=3') == mime.GML3
    assert mime.get('application/gml+xml; version=3') == mime.GML3


def test_common_contains_constants():
    assert mime.get(mime.GPKG) == mime.GPKG
    assert mime.get(mime.GML32) == mime.GML32
    assert mime.get(mime.MULTIPART) == mime.MULTIPART


def test_get_ows_aliases():
    assert mime.get('application/vnd.ogc.wms_xml') == mime.XML
    assert mime.get('application/vnd.ogc.se_xml') == mime.XML
    assert mime.get('text/xml; subtype=gml/3.1.1') == mime.GML3
    assert mime.get('application/vnd.ogc.gml/3.1.1') == mime.GML3
    assert mime.get('application/json; subtype=geojson') == mime.GEOJSON


def test_get_new_aliases():
    assert mime.get('image/svg') == mime.SVG
    assert mime.get('application/javascript') == mime.JS
    assert mime.get('application/x-font-ttf') == mime.TTF
    assert mime.get('image/geotiff') == mime.TIFF
    assert mime.get('tif') == mime.TIFF
