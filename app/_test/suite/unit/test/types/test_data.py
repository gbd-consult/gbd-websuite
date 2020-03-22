"""Test the types.Data object"""

import gws.core.util as cu
import gws.types as t

import _test.util as u


def test_init():
    d = t.Data(
        {
            'dict1': 'DICT1',
            'dict2': 'DICT2',
        },
        t.Data(data1='DATA1', data2='DATA2'),
        kw1='KW1',
        kw2='KW2'
    )

    assert vars(d) == {
        'data1': 'DATA1',
        'data2': 'DATA2',
        'dict1': 'DICT1',
        'dict2': 'DICT2',
        'kw1': 'KW1',
        'kw2': 'KW2'}


def test_get():
    d = t.Data(a='A', b='B')

    assert d.get('a') == 'A'
    assert d.get('ZERO') is None
    assert d.get('ZERO', 'miss') == 'miss'


def test_attr_handler():
    d = t.Data(a='A', b='B')

    assert d.a == 'A'
    assert d.ZERO is None

    with u.raises(AttributeError):
        a = d._special
