import os
import gws.config
import gws.base.feature
import gws.lib.sa as sa
import gws.test.util as u


@u.fixture(scope='module')
def gws_root():
    u.pg_create_table(
        'plain',
        {'id': 'int primary key', 'str1': 'text', 'str2': 'text'},
        {'id': 1, 'str1': 'abc 1', 'str2': 'uvw 1'},
        {'id': 2, 'str1': 'def 2', 'str2': 'xyz 2'},
        {'id': 3, 'str1': 'abc 3', 'str2': 'uvw 3'},
        {'id': 4, 'str1': 'def 4', 'str2': 'xyz 4'},
    )
    cfg = '''
        database.providers+ { type "postgres" serviceName "gws_test_postgres" } 
        
        models+ { 
            uid "NO_SEARCH" type "postgres" tableName "plain" 
            fields+ { name "str1" type text } 
        }
        models+ { 
            uid "EXACT_1" type "postgres" tableName "plain"
            fields+ { name "str1" type text textSearch.type exact } 
        }
        models+ { 
            uid "ANY_1" type "postgres" tableName "plain"
            fields+ { name "str1" type text textSearch.type any } 
        }
        models+ { 
            uid "ANY_1_OR_2" type "postgres" tableName "plain"
            fields+ { name "str1" type text textSearch.type any } 
            fields+ { name "str2" type text textSearch.type any } 
        }
        models+ { 
            uid "BEGIN_1" type "postgres" tableName "plain"
            fields+ { name "str1" type text textSearch.type begin } 
        }
        models+ { 
            uid "END_1" type "postgres" tableName "plain"
            fields+ { name "str1" type text textSearch.type begin } 
        }
    '''

    yield u.gws_configure(cfg)


def _search(**kwargs):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        search=gws.SearchQuery(**kwargs),
        op=gws.ModelOperation.read,
    )
    return mc


##

def test_no_search_no_results(gws_root):
    mm = u.model(gws_root, 'NO_SEARCH')

    fs = mm.find_features(_search(keyword='abc'))
    assert u.model_feature_atts(fs, 'str1') == []


def test_exact_search(gws_root):
    mm = u.model(gws_root, 'EXACT_1')

    fs = mm.find_features(_search(keyword='abc'))
    assert u.model_feature_atts(fs, 'str1') == []

    fs = mm.find_features(_search(keyword='abc 1'))
    assert u.model_feature_atts(fs, 'str1') == ['abc 1']


def test_any_search(gws_root):
    mm = u.model(gws_root, 'ANY_1')

    fs = mm.find_features(_search(keyword='boo'))
    assert u.model_feature_atts(fs, 'str1') == []

    fs = mm.find_features(_search(keyword='abc'))
    assert u.model_feature_atts(fs, 'str1') == ['abc 1', 'abc 3']
