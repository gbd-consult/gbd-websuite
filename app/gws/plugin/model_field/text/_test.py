import gws.config
import gws.base.feature
import gws.test.util as u


@u.fixture(scope='module')
def root():
    u.pg.create('plain', {'id': 'int primary key', 'a': 'text', 'b': 'text'})
    u.pg.insert('plain', [
        {'id': 1, 'a': 'abc 1', 'b': 'uvw 1'},
        {'id': 2, 'a': 'def 2', 'b': 'xyz 2'},
        {'id': 3, 'a': 'abc 3', 'b': 'uvw 3'},
        {'id': 4, 'a': 'def 4', 'b': 'xyz 4'},
        {'id': 5, 'a': 'a%c 5', 'b': ''},
        {'id': 6, 'a': 'a_c 6', 'b': ''},
        {'id': 7, 'a': 'a\\c 7', 'b': ''},
    ])
    cfg = '''
        models+ { 
            uid "NO_SEARCH" type "postgres" tableName "plain" 
            fields+ { name "a" type text } 
        }
        models+ { 
            uid "EXACT_1" type "postgres" tableName "plain"
            fields+ { name "a" type text textSearch.type exact } 
        }
        models+ { 
            uid "ANY_1" type "postgres" tableName "plain"
            fields+ { name "a" type text textSearch.type any } 
        }
        models+ { 
            uid "ANY_1_OR_2" type "postgres" tableName "plain"
            fields+ { name "a" type text textSearch.type any } 
            fields+ { name "b" type text textSearch.type any } 
        }
        models+ { 
            uid "BEGIN_1" type "postgres" tableName "plain"
            fields+ { name "a" type text textSearch.type begin } 
        }
        models+ { 
            uid "END_1" type "postgres" tableName "plain"
            fields+ { name "a" type text textSearch.type end } 
        }
        models+ { 
            uid "ANY_CASE_1" type "postgres" tableName "plain"
            fields+ { name "a" type text textSearch { type any caseSensitive true } } 
        }
    '''

    yield u.gws_root(cfg)


##

def test_no_search_no_results(root: gws.Root):
    mm = u.cast(gws.Model, root.get('NO_SEARCH'))

    fs = mm.find_features(gws.SearchQuery(keyword='abc'), u.model.context())
    assert [f.get('a') for f in fs] == []


def test_exact_search(root: gws.Root):
    mm = u.cast(gws.Model, root.get('EXACT_1'))

    fs = mm.find_features(gws.SearchQuery(keyword='abc'), u.model.context())
    assert [f.get('a') for f in fs] == []

    fs = mm.find_features(gws.SearchQuery(keyword='abc 1'), u.model.context())
    assert [f.get('a') for f in fs] == ['abc 1']


def test_any_search(root: gws.Root):
    mm = u.cast(gws.Model, root.get('ANY_1'))

    fs = mm.find_features(gws.SearchQuery(keyword='boo'), u.model.context())
    assert [f.get('a') for f in fs] == []

    fs = mm.find_features(gws.SearchQuery(keyword='abc'), u.model.context())
    assert [f.get('a') for f in fs] == ['abc 1', 'abc 3']


def _search(root, model_uid, keyword):
    mm = u.cast(gws.Model, root.get(model_uid))
    fs = mm.find_features(gws.SearchQuery(keyword=keyword), u.model.context())
    return sorted(f.get('a') for f in fs)


def test_begin_search(root: gws.Root):
    assert _search(root, 'BEGIN_1', 'abc') == ['abc 1', 'abc 3']
    assert _search(root, 'BEGIN_1', '1') == []


def test_end_search(root: gws.Root):
    assert _search(root, 'END_1', '1') == ['abc 1']
    assert _search(root, 'END_1', 'abc') == []


def test_any_search_is_case_insensitive(root: gws.Root):
    assert _search(root, 'ANY_1', 'ABC') == ['abc 1', 'abc 3']


def test_any_search_case_sensitive(root: gws.Root):
    assert _search(root, 'ANY_CASE_1', 'ABC') == []
    assert _search(root, 'ANY_CASE_1', 'abc') == ['abc 1', 'abc 3']


def test_wildcards_are_literal(root: gws.Root):
    assert _search(root, 'ANY_1', '%') == ['a%c 5']
    assert _search(root, 'ANY_1', '_') == ['a_c 6']
    assert _search(root, 'ANY_1', '\\') == ['a\\c 7']
    assert _search(root, 'BEGIN_1', 'a%') == ['a%c 5']
    assert _search(root, 'END_1', '_c 6') == ['a_c 6']
