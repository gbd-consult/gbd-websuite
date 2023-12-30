from gws.core.data import Data
import gws.test.util as u


def test_init_accepts_dicts_and_kwargs():
    d = Data({'a': 'A', 'b': 'B'}, c='C', d='D')
    assert vars(d) == {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'}


def test_init_accepts_data_objects():
    e = Data(a='A', b='B')
    d = Data(e, c='C')
    assert vars(d) == {'a': 'A', 'b': 'B', 'c': 'C'}


def test_unknown_props_return_none():
    d = Data({'a': 'A', 'b': 'B'})
    # noinspection PyUnresolvedReferences
    assert d.xxx is None


def test_unknown_private_props_throw():
    d = Data({'a': 'A', 'b': 'B'})
    with test.raises(AttributeError):
        # noinspection PyUnresolvedReferences
        assert d._xxx is None

