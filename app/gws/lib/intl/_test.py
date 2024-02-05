"""Tests for the intl module."""

import gws
import gws.test.util as u
import gws.lib.intl as intl
import re

de = {'id': 'de_DE',
      'dateFormatLong': 'd. MMMM y',
      'dateFormatMedium': 'dd.MM.y',
      'dateFormatShort': 'dd.MM.yy',
      'dateUnits': 'JMT',
      'dayNamesLong': ['Sonntag', 'Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag'],
      'dayNamesShort': ['So.', 'Mo.', 'Di.', 'Mi.', 'Do.', 'Fr.', 'Sa.'],
      'dayNamesNarrow': ['S', 'M', 'D', 'M', 'D', 'F', 'S'],
      'firstWeekDay': 0,
      'language': 'de',
      'languageName': 'Deutsch',
      'monthNamesLong': ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober',
                         'November', 'Dezember'],
      'monthNamesShort': ['Jan.', 'Feb.', 'März', 'Apr.', 'Mai', 'Juni', 'Juli', 'Aug.', 'Sept.', 'Okt.', 'Nov.',
                          'Dez.'],
      'monthNamesNarrow': ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'],
      'numberDecimal': ',',
      'numberGroup': '.'}


def test_locale():
    assert intl.locale('de').__dict__ == de
    assert intl.locale('de-DE').__dict__ == de
    assert intl.locale('de_DE').__dict__ == de


def test_bibliographic_name():
    assert intl.bibliographic_name('en') == 'eng'
    assert intl.bibliographic_name('de') == 'deu'
    assert intl.bibliographic_name('es') == 'spa'


patterniso = r'\d{4}-((0\d)|(1[0-2]))-(([0-2]\d)|(3[01]))T(([0-1]\d)|(2[0-3])):[0-5]\d:[0-5]\d.\d{6}'


def test_date_format():
    d = intl.date_formatter('de_DE')
    short = d.format(intl.DateTimeFormat.short)
    patterns = r'(([0-2]\d)|(3[01])).((0\d)|(1[0-2])).\d{2}'  # 01.02.24
    assert re.match(r'^' + patterns + r'$', short)
    medium = d.format(intl.DateTimeFormat.medium)
    patternm = r'(([0-2]\d)|(3[01])).((0\d)|(1[0-2])).\d{4}'  # 01.02.2024
    assert re.match(r'^' + patternm + r'$', medium)
    long = d.format(intl.DateTimeFormat.long)
    patternl = r'((\d)|([12]\d)|(3[01])). ((Januar)|(Februar)|(März)|(April)|(Mai)|(Juni)|(Juli)|(August)|(September)|(Oktober)|(November)|(Dezember)) \d{4}'  # 1. Februar 2024
    assert re.match(r'^' + patternl + r'$', long)
    iso = d.format(intl.DateTimeFormat.iso)
    patterni = patterniso  # 2024-02-01T13:31:57.164771
    assert re.match(patterni, iso)


def test_time_format():
    d = intl.time_formatter('de_DE')
    short = d.format(intl.DateTimeFormat.short)
    patterns = r'(([0-1]\d)|(2[0-3])):[0-5]\d'  # 15:40
    assert re.match(r'^' + patterns + r'$', short)
    medium = d.format(intl.DateTimeFormat.medium)
    patternm = patterns + r':[0-5]\d'  # 13:32:53
    assert re.match(r'^' + patternm + r'$', medium)
    long = d.format(intl.DateTimeFormat.long)
    patternl = patternm + r' UTC'  # 13:33:03 UTC
    assert re.match(r'^' + patternl + r'$', long)
    iso = d.format(intl.DateTimeFormat.iso)
    patterni = patterniso  # 2024-02-01T13:33:16.566305
    assert re.match(r'^' + patterni + r'$', iso)


def test_number_format():
    d = intl.number_formatter('de')
    assert d.format(intl.NumberFormat.decimal, 00012345678.90000) == '12345678,9'
    assert d.format(intl.NumberFormat.grouped, 00012345678.90000) == '12.345.678,9'
    assert d.format(intl.NumberFormat.currency, 00012345678.9000, currency='EUR') == '12.345.678,90 €'
    assert d.format(intl.NumberFormat.percent, 00012345678.90000) == '1.234.567.890 %'


def test_date_formatter():
    gws.delete_app_global('gws.lib.intl.number_formatter.de-DE')
    d = intl.date_formatter('de-DE')
    assert intl.date_formatter('de-DE') == d


def test_time_formatter():
    gws.delete_app_global('gws.lib.intl.number_formatter.en-GB')
    t = intl.time_formatter('en-GB')
    assert intl.time_formatter('en-GB') == t


def test_number_formatter():
    gws.delete_app_global('gws.lib.intl.number_formatter.es-SP')
    n = intl.number_formatter('es-SP')
    assert intl.number_formatter('es-SP') == n
