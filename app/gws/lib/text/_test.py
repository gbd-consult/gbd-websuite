"""Tests for the text module."""

import gws
import gws.lib.text as text
import gws.test.util as u


def test_to_base64():
    assert text.to_base64('hello') == 'aGVsbG8='
    assert text.to_base64(b'\xfb\xff') == '+/8='


def test_to_base64_url_safe():
    assert text.to_base64(b'\xfb\xff', url_safe=True) == '-_8='


def test_to_base64_no_line_breaks():
    assert '\n' not in text.to_base64('x' * 200)


def test_from_base64():
    assert text.from_base64('aGVsbG8=') == b'hello'
    assert text.from_base64(b'+/8=') == b'\xfb\xff'


def test_from_base64_url_safe():
    assert text.from_base64('-_8=', url_safe=True) == b'\xfb\xff'


def test_from_base64_invalid():
    with u.raises(text.Error):
        text.from_base64('aGVs!bG8=')
    with u.raises(text.Error):
        text.from_base64('-_8=')
    with u.raises(text.Error):
        text.from_base64('aGVsbG8')


def test_data_url_roundtrip():
    url = text.to_data_url(b'<svg/>', 'image/svg+xml')
    assert url == 'data:image/svg+xml;base64,PHN2Zy8+'
    assert text.parse_data_url(url) == ('image/svg+xml', b'<svg/>')


def test_parse_data_url_percent_encoded():
    assert text.parse_data_url('data:image/svg+xml;utf8,%3Csvg%2F%3E') == ('image/svg+xml', b'<svg/>')


def test_parse_data_url_params():
    assert text.parse_data_url('data:Text/HTML;charset=utf8;base64,aGk=') == ('text/html', b'hi')


def test_parse_data_url_default_mime_type():
    assert text.parse_data_url('data:,hi') == ('text/plain', b'hi')


def test_parse_data_url_invalid():
    with u.raises(text.Error):
        text.parse_data_url('http://example.com')
    with u.raises(text.Error):
        text.parse_data_url('data:image/png;base64,!!!')


def test_dedent():
    assert text.dedent('\n    a\n      b\n    c\n') == '\na\n  b\nc\n'


def test_to_lines():
    assert text.to_lines(' a \n\n b # comment\n# only comment\n', comment='#') == ['a', 'b']


def test_to_int_str():
    assert text.to_int_str(12.7) == '12'
    assert text.to_int_str(3) == '3'
