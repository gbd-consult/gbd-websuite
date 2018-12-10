from .. import util


def test_get_with_simple_prop():
    d = {'alef': 'ok'}
    assert util.get(d, 'alef') == 'ok'
    assert util.get(d, 'ZERO', 'miss') == 'miss'


def test_get_with_compound_prop():
    d = {
        'alef': {
            'bet': {
                'gimel': 'ok'
            },
            'dalet': 'x',
            'he': 'y'
        }
    }
    assert util.get(d, 'alef.bet.gimel') == 'ok'
    assert util.get(d, 'alef.ZERO.gimel', 'miss') == 'miss'


def test_get_with_list_index():
    d = {
        'alef': {
            'bet': ['gimel', 'dalet', {'he': 'ok'}, 'vav']
        }
    }
    assert util.get(d, 'alef.bet[2].he') == 'ok'
    assert util.get(d, 'alef.bet[9].he', 'miss') == 'miss'


def test_get_with_list_implicit_zero():
    d = {
        'alef': {
            'bet': [{'gimel': 'ok'}, 'dalet'],
            'he': []
        }
    }
    assert util.get(d, 'alef.bet.gimel') == 'ok'
    assert util.get(d, 'alef.he.gimel', 'miss') == 'miss'


def test_get_with_object():
    class T:
        def __init__(self):
            self.bet = 'ok'

    d = {'alef': T()}
    assert util.get(d, 'alef.bet') == 'ok'
    assert util.get(d, 'alef.ZERO', 'miss') == 'miss'


#####

def test_extend():
    a = {'alef': 1, 'bet': 2}
    b = {'bet': 22, 'gimel': 33}

    assert util.extend(a, b, gimel=999) == {'alef': 1, 'bet': 22, 'gimel': 999}
    assert util.defaults(a, b, gimel=99) == {'alef': 1, 'bet': 2, 'gimel': 33}

#####

def test_is_empty():
    class T:
        def __init__(self, x=None):
            if x:
                self.x = x

    assert util.is_empty(None)
    assert util.is_empty([])
    assert util.is_empty('')
    assert util.is_empty(T())

    assert not util.is_empty([''])
    assert not util.is_empty('  ')
    assert not util.is_empty(333)
    assert not util.is_empty(T(333))


#####

def test_compact_strip():
    d = {'alef': 'A', 'ZERO1': None, 'bet': ' B ', 'ZERO2': None, 'gimel': [], 'dalet': '   '}
    assert util.compact(d) == {'alef': 'A', 'bet': ' B ', 'gimel': [], 'dalet': '   '}
    assert util.strip(d) == {'alef': 'A', 'bet': 'B'}

    d = ['alef', None, 'bet', None, 'gimel']
    assert util.compact(x for x in d) == ['alef', 'bet', 'gimel']

#####
