import gws
import gws.test.util as u


def test_init_accepts_dicts_and_kwargs():
    d = gws.Data({'a': 'A', 'b': 'B'}, c='C', d='D')
    assert vars(d) == {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'}


def test_init_accepts_data_objects():
    e = gws.Data(a='A', b='B')
    d = gws.Data(e, c='C')
    assert vars(d) == {'a': 'A', 'b': 'B', 'c': 'C'}


def test_unknown_props_return_none():
    d = gws.Data({'a': 'A', 'b': 'B'})
    assert d.xxx is None


def test_unknown_private_props_throw():
    d = gws.Data({'a': 'A', 'b': 'B'})
    with u.raises(AttributeError):
        assert d._xxx is None

