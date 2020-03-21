import gws.core.util as cu
import gws.types as t

import _test.util as u


def test_get():
    d = {'a': 'ok'}

    assert cu.get(d, 'a') == 'ok'
    assert cu.get(d, 'ZERO', 'miss') == 'miss'
    assert cu.get(d, 'ZERO') is None

    d = {
        'a': {
            'b': {
                'c': 'ok'
            },
            'd': 'x',
            'e': 'y'
        }
    }

    assert cu.get(d, 'a.b.c') == 'ok'
    assert cu.get(d, 'a.ZERO.d', 'miss') == 'miss'

    d = {
        'a': {
            'b': ['c', 'd', {'e': 'ok'}, 'f']
        }
    }

    assert cu.get(d, 'a.b.2.e') == 'ok'
    assert cu.get(d, 'a.b.9.f', 'miss') == 'miss'

    d = {'a': t.Data(b=[0, 'ok'])}

    assert cu.get(d, 'a.b.1') == 'ok'
    assert cu.get(d, 'a.ZERO', 'miss') == 'miss'

    d = [0, 1, {'a': 'ok'}]

    assert cu.get(d, '2.a') == 'ok'
    assert cu.get(d, 'ZERO', 'miss') == 'miss'



def test_extend():
    a = {'a': 1, 'b': 2}
    b = {'b': 22, 'c': 33}

    assert cu.merge(a, b, c=77, e=777) == {'a': 1, 'b': 22, 'c': 77, 'e': 777}

    a = t.Data(a)
    e = cu.merge(a, b, c=77, e=777)

    assert isinstance(e, t.Data)
    assert vars(e) == {'a': 1, 'b': 22, 'c': 77, 'e': 777}


def test_filter():
    d = {'a': 'A', 'b': None, 'c': '', 'd': {}, 'e': [], 'f': t.Data(), 'g': '   '}

    assert cu.compact(d) == {'a': 'A', 'c': '', 'd': {}, 'e': [], 'f': d['f'], 'g': '   '}
    assert cu.filter(d) == {'a': 'A'}
    assert cu.filter(d, lambda x: isinstance(x, str)) == {'a': 'A', 'c': '', 'g': '   '}
    assert cu.strip(d) == {'a': 'A'}

    e = t.Data(d)

    assert vars(cu.compact(e)) == {'a': 'A', 'c': '', 'd': {}, 'e': [], 'f': d['f'], 'g': '   '}
    assert vars(cu.filter(e)) == {'a': 'A'}
    assert vars(cu.filter(e, lambda x: isinstance(x, str))) == {'a': 'A', 'c': '', 'g': '   '}
    assert vars(cu.strip(e)) == {'a': 'A'}


def test_map():
    d = {'a': 'A', 'b': 'B'}

    assert cu.map(d, str.lower) == {'a': 'a', 'b': 'b'}



def test_is_empty():
    class T:
        def __init__(self, x=None):
            if x:
                self.x = x

    assert cu.is_empty(None)
    assert cu.is_empty([])
    assert cu.is_empty('')
    assert cu.is_empty(T())

    assert not cu.is_empty([''])
    assert not cu.is_empty('  ')
    assert not cu.is_empty(333)
    assert not cu.is_empty(T(333))
