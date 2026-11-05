"""Tests for the cql params module."""

import pytest

import gws.lib.cql as cql


@pytest.mark.parametrize(
    'value,expected',
    [
        ('2021-06-15T10:00:00Z', '"a" = TIMESTAMP(\'2021-06-15T10:00:00Z\')'),
        ('2021-06-15', '"a" = DATE(\'2021-06-15\')'),
        ('2021-06-01/2021-06-30', '"a" >= DATE(\'2021-06-01\') AND "a" <= DATE(\'2021-06-30\')'),
        ('2021-06-01T00:00:00Z/..', '"a" >= TIMESTAMP(\'2021-06-01T00:00:00Z\')'),
        ('2021-06-01T00:00:00Z/', '"a" >= TIMESTAMP(\'2021-06-01T00:00:00Z\')'),
        ('../2021-06-30', '"a" <= DATE(\'2021-06-30\')'),
        ('/2021-06-30', '"a" <= DATE(\'2021-06-30\')'),
        ('2021-06-01/2021-06-30T12:00:00+02:00', '"a" >= DATE(\'2021-06-01\') AND "a" <= TIMESTAMP(\'2021-06-30T12:00:00+02:00\')'),
        ('../..', ''),
        ('/', ''),
        ('', ''),
    ],
)
def test_datetime_param(value, expected):
    assert cql.from_datetime_param(value, 'a') == expected


def test_datetime_param_quotes_the_name():
    assert cql.from_datetime_param('2021-06-15', 'a"b') == '"a""b" = DATE(\'2021-06-15\')'


def test_datetime_param_parses():
    src = cql.from_datetime_param('2021-06-01/2021-06-30T12:00:00Z', 'a')
    assert cql.parse(src)[0] == cql.Node.AND


@pytest.mark.parametrize(
    'value',
    [
        'nonesuch',
        '..',
        '2021-13-45',
        '2021-06-01/nonesuch',
        '2021-06-01/2021-06-02/2021-06-03',
        "2021-06-01') OR (1 = 1",
    ],
)
def test_invalid_datetime_param(value):
    with pytest.raises(cql.ParseError):
        cql.from_datetime_param(value, 'a')
