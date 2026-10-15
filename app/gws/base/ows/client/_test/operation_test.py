"""Tests for the operation module."""

import gws
import gws.test.util as u
import gws.base.ows.client.operation as operation


def _create_from_caps(mime_types):
    return operation.create(gws.OwsVerb.GetMap, gws.OwsOperation(mimeTypes=mime_types))


def test_create_normalizes_caps_mime_types():
    op = _create_from_caps(['Image/PNG', 'image/png;mode=8bit', 'image/png; mode=8bit'])
    assert op.mimeTypes == ['image/png', 'image/png; mode=8bit']
    assert op.mimeTypeMap == {'image/png': 'Image/PNG', 'image/png; mode=8bit': 'image/png;mode=8bit'}


def test_create_resolves_aliases():
    op = _create_from_caps(['image/png8', 'image/png; mode=8bit', 'image/jpg'])
    assert op.mimeTypes == ['image/png; mode=8bit', 'image/jpeg']
    assert op.mimeTypeMap == {'image/png; mode=8bit': 'image/png8', 'image/jpeg': 'image/jpg'}


def test_create_ignores_unknown_mime_types():
    op = _create_from_caps(['foo/bar', 'image/png'])
    assert op.mimeTypes == ['image/png']


def test_create_same_spelling_has_empty_mime_type_map():
    op = _create_from_caps(['image/png', 'image/png; mode=8bit'])
    assert op.mimeTypeMap == {}


def test_create_default_from_caps():
    op = _create_from_caps(['image/jpeg', 'image/png8'])
    assert op.defaultMimeType == 'image/png; mode=8bit'


def test_create_default_without_mime_types():
    op = operation.create(gws.OwsVerb.GetMap)
    assert op.mimeTypes == []
    assert op.defaultMimeType == 'image/png'


def test_create_config_formats_first_is_default():
    cfg = operation.Config(verb=gws.OwsVerb.GetMap, formats=['image/jpeg', 'image/png'])
    op = operation.create(gws.OwsVerb.GetMap, None, [cfg])
    assert op.mimeTypes == ['image/jpeg', 'image/png']
    assert op.defaultMimeType == 'image/jpeg'


def test_create_config_empty_formats():
    cfg = operation.Config(verb=gws.OwsVerb.GetMap, formats=[])
    op = operation.create(gws.OwsVerb.GetMap, None, [cfg])
    assert op.mimeTypes == []
    assert op.defaultMimeType == 'image/png'


def test_create_config_overrides_caps():
    co = gws.OwsOperation(url='url_1', params={'a': '1'})
    cfg = operation.Config(verb=gws.OwsVerb.GetMap, url='url_2')
    op = operation.create(gws.OwsVerb.GetMap, co, [cfg])
    assert op.url == 'url_2'
    assert op.params == {'a': '1'}


def test_create_default_by_base_type():
    op = _create_from_caps(['image/gif', 'image/png; mode=16bit'])
    assert op.defaultMimeType == 'image/png; mode=16bit'


def test_create_default_first_supported():
    op = _create_from_caps(['image/gif', 'image/tiff'])
    assert op.defaultMimeType == 'image/gif'


def test_create_default_for_feature_requests():
    co = gws.OwsOperation(mimeTypes=['text/html', 'application/vnd.ogc.gml', 'text/plain'])
    op = operation.create(gws.OwsVerb.GetFeatureInfo, co)
    assert op.defaultMimeType == gws.lib.mime.GML2
    assert op.mimeTypeMap[op.defaultMimeType] == 'application/vnd.ogc.gml'


def test_create_default_prefers_gml3():
    co = gws.OwsOperation(mimeTypes=['application/gml+xml; version=3.2', 'text/xml; subtype=gml/3.1.1'])
    op = operation.create(gws.OwsVerb.GetFeature, co)
    assert op.defaultMimeType == gws.lib.mime.GML3


def test_create_default_for_feature_requests_without_mime_types():
    op = operation.create(gws.OwsVerb.GetFeature)
    assert op.defaultMimeType == gws.lib.mime.GML


def test_create_config_for_other_verb_ignored():
    cfg = operation.Config(verb=gws.OwsVerb.GetFeatureInfo, url='url_2')
    op = operation.create(gws.OwsVerb.GetMap, gws.OwsOperation(url='url_1'), [cfg])
    assert op.url == 'url_1'
