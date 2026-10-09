"""Tests for the style module."""

import base64
import urllib.parse

import gws
import gws.lib.style as style
import gws.test.util as u

_test_against_dict = {
    'fill': None,
    'icon': None,
    'label_align': 'center',
    'label_font_family': 'sans-serif',
    'label_font_size': 12,
    'label_font_style': 'normal',
    'label_font_weight': 'normal',
    'label_line_height': 1,
    'label_max_scale': 1000000000,
    'label_min_scale': 0,
    'label_offset_x': None,
    'label_offset_y': None,
    'label_placement': 'middle',
    'label_stroke_dasharray': [],
    'label_stroke_dashoffset': 0,
    'label_stroke_linecap': 'butt',
    'label_stroke_linejoin': 'miter',
    'label_stroke_miterlimit': 2222,
    'label_stroke_width': 0,
    'marker_fill': 'rgb(255,90,33)',
    'marker_size': 0,
    'marker_stroke_dasharray': [],
    'marker_stroke_dashoffset': 0,
    'marker_stroke_linecap': 'butt',
    'marker_stroke_linejoin': 'miter',
    'marker_stroke_miterlimit': 0,
    'marker_stroke_width': 0,
    'offset_x': 0,
    'offset_y': 0,
    'point_size': 10,
    'stroke': None,
    'stroke_dasharray': [],
    'stroke_dashoffset': 0,
    'stroke_linecap': 'butt',
    'stroke_linejoin': 'miter',
    'stroke_miterlimit': 0,
    'stroke_width': 0,
    'with_geometry': 'all',
    'with_label': 'all',
}

_values = {'fill': None,
           'stroke': None,
           'stroke_dasharray': [],
           'stroke_dashoffset': 0,
           'stroke_linecap': 'butt',
           'stroke_linejoin': 'miter',
           'stroke_miterlimit': 0,
           'stroke_width': 0,
           'marker_size': 0,
           'marker_stroke_dasharray': [],
           'marker_stroke_dashoffset': 0,
           'marker_stroke_linecap': 'butt',
           'marker_stroke_linejoin': 'miter',
           'marker_stroke_miterlimit': 0,
           'marker_stroke_width': 0,
           'with_geometry': 'all',
           'with_label': 'all',
           'label_align': 'center',
           'label_font_family': 'sans-serif',
           'label_font_size': 12,
           'label_font_style': 'normal',
           'label_font_weight': 'normal',
           'label_line_height': 1,
           'label_max_scale': 1000000000,
           'label_min_scale': 0,
           'label_offset_x': None,
           'label_offset_y': None,
           'label_placement': 'middle',
           'label_stroke_dasharray': [],
           'label_stroke_dashoffset': 0,
           'label_stroke_linecap': 'butt',
           'label_stroke_linejoin': 'miter',
           'label_stroke_miterlimit': 0,
           'label_stroke_width': 0,
           'point_size': 10,
           'icon': None,
           'offset_x': 0, 'offset_y': 0,
           'marker_fill': 'rgb(255,90,33)',
           'label_fill': 'foo'}

_test_against_obj = style.core.Object('',
                                      '__label-stroke_miterlimit: 2222; marker_fill: rgb(255, 90, 33)',
                                      _values)


# tests for core
def test_from_dict():
    dic = {'text': '__label-stroke_miterlimit: 2222; marker_fill: rgb(255, 90, 33)',
           'values': {'with_geometry': 'ba', 'label_fill': 'foo'}}

    opt = style.parser.Options(trusted=True, strict=True, imageDirs=())
    ret = style.core.from_dict(dic, opt)
    ret_dic = ret.__dict__
    assert ret_dic.__str__() == _test_against_obj.__dict__.__str__()


def test_from_config():
    opt = style.parser.Options(trusted=True, strict=True, imageDirs=())
    cfg = style.core.Config()
    cfg.set('cssSelector', '')
    cfg.set('text', '__label-stroke_miterlimit: 2222; marker_fill: rgb(255, 90, 33)')
    cfg.set('values', {'with_geometry': 'ba', 'label_fill': 'foo'})
    ret = style.core.from_config(cfg, opt)
    ret_dic = ret.__dict__
    assert ret_dic.__str__() == _test_against_obj.__dict__.__str__()


def test_from_props():
    opt = style.parser.Options(trusted=True, strict=True, imageDirs=())
    p = style.core.Props()
    p.set('cssSelector', '')
    _test_against_obj.text = ''
    _values['label_stroke_miterlimit'] = 0
    _values.pop('marker_fill')
    _test_against_obj.values = _values
    p.set('values', {'with_geometry': 'ba', 'label_fill': 'foo'})
    ret = style.core.from_props(p, opt)
    ret_dic = ret.__dict__
    assert ret_dic.__str__() == _test_against_obj.__dict__.__str__()


def test_to_data_url_empty():
    assert style.icon.to_data_url(None) == ''


# tests for icon
def test_icon():
    url = 'https://mdn.dev/archives/media/attachments/2012/07/09/3075/89b1e0a26e8421e19f907e0522b188bd/svgdemo1.xml'
    opt = style.parser.Options(trusted=True, strict=True, imageDirs=())
    icon = style.icon.parse(url, opt)
    url2 = style.icon.to_data_url(icon)
    icon2 = style.icon.parse(url2, opt)
    assert icon2.to_dict() == icon.to_dict()


def test_parse_untrusted():
    url = 'https://mdn.dev/archives/media/attachments/2012/07/09/3075/89b1e0a26e8421e19f907e0522b188bd/svgdemo1.xml'
    opt = style.parser.Options(trusted=False, strict=True, imageDirs=())
    with u.raises(Exception):
        style.icon.parse(url, opt)


_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><circle cx="5" cy="5" r="4"/></svg>'


def test_parse_data_url_utf8():
    opt = style.parser.Options(trusted=False, strict=True, imageDirs=())
    icon = style.icon.parse('data:image/svg+xml;utf8,' + _SVG, opt)
    assert icon.get('width') == '10'


def test_parse_data_url_utf8_percent_encoded():
    opt = style.parser.Options(trusted=False, strict=True, imageDirs=())
    icon = style.icon.parse('data:image/svg+xml;utf8,' + urllib.parse.quote(_SVG), opt)
    assert icon.get('width') == '10'


def test_parse_data_url_base64():
    opt = style.parser.Options(trusted=False, strict=True, imageDirs=())
    icon = style.icon.parse('data:image/svg+xml;base64,' + base64.b64encode(_SVG.encode('utf8')).decode('utf8'), opt)
    assert icon.get('width') == '10'


def test_parse_data_url_invalid_base64():
    opt = style.parser.Options(trusted=False, strict=True, imageDirs=())
    with u.raises(style.icon.Error):
        style.icon.parse('data:image/svg+xml;base64,PHN2Zz4!!!', opt)


# tests for parser
def test_parse_dict():
    dic = {'__label-stroke_miterlimit': 2222,
           'marker_fill': 'rgb(255, 90, 33)', }
    opt = style.parser.Options(trusted=True, strict=True, imageDirs=())
    assert style.parser.parse_dict(dic, opt) == _test_against_dict


def test_parse_dict_error():
    dic = {'__label-stroke_miterlimit': 2222,
           'marker_fill': 'fo',
           'with_geometry': 'ba',
           'label_fill': 'fo'}
    opt = style.parser.Options(trusted=True, strict=True, imageDirs=())
    with u.raises(Exception):
        assert style.parser.parse_dict(dic, opt)


def test_parse_text():
    text = ('__label-stroke_miterlimit: 2222; '
            ';'
            #         ' :  ;'
            'marker_fill: rgb(255, 90, 33)')
    opt = style.parser.Options(trusted=True, strict=True, imageDirs=())
    assert style.parser.parse_text(text, opt) == _test_against_dict
