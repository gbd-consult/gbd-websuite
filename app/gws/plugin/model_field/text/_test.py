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
            fields+ { name "a" type text textSearch.type begin } 
        }
    '''

    yield u.gws_root(cfg)


##

def test_no_search_no_results(root: gws.Root):
    mm = u.cast(gws.Model, root.get('NO_SEARCH'))

    fs = mm.find_features(gws.SearchQuery(keyword='abc'), u.gws_model_context())
    assert [f.get('a') for f in fs] == []


def test_exact_search(root: gws.Root):
    mm = u.cast(gws.Model, root.get('EXACT_1'))

    fs = mm.find_features(gws.SearchQuery(keyword='abc'), u.gws_model_context())
    assert [f.get('a') for f in fs] == []

    fs = mm.find_features(gws.SearchQuery(keyword='abc 1'), u.gws_model_context())
    assert [f.get('a') for f in fs] == ['abc 1']


def test_any_search(root: gws.Root):
    mm = u.cast(gws.Model, root.get('ANY_1'))

    fs = mm.find_features(gws.SearchQuery(keyword='boo'), u.gws_model_context())
    assert [f.get('a') for f in fs] == []

    fs = mm.find_features(gws.SearchQuery(keyword='abc'), u.gws_model_context())
    assert [f.get('a') for f in fs] == ['abc 1', 'abc 3']
