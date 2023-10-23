import os
import gws.config
import gws.base.feature
import gws.lib.sa as sa
import gws.lib.test.util as u
import gws.types as t


def test_database_is_up():
    res = u.pg_connect().execute(sa.text('select 42'))
    assert res.scalar_one() == 42


##


@u.fixture(scope='module')
def gws_root():
    u.pg_create_table(
        'plain',
        {'id': 'int primary key', 'str1': 'text'},
        {'id': 1, 'str1': '11'},
        {'id': 2, 'str1': '22'},
        {'id': 3, 'str1': '33'},
        {'id': 4, 'str1': '44'},
    )
    u.pg_create_table(
        'serial_id',
        {'id': 'serial primary key', 'str1': 'text', 'str2': 'text'},
    )

    cfg = '''
        database.providers+ { type "postgres" serviceName "gws_test_postgres" } 
        
        models+ { 
            uid "PLAIN" type "postgres" tableName "plain" 
        }
        models+ { 
            uid "SERIAL_ID" type "postgres" tableName "serial_id"
        }
    '''

    yield u.gws_configure(cfg)


##

def test_find_by_ids(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        op=gws.ModelOperation.read,
    )
    mm = u.model(gws_root, 'PLAIN')
    fs = mm.find_features(gws.SearchQuery(uids=[2, 3]), mc)
    assert u.model_feature_atts(fs, 'str1') == ['22', '33']


##

def test_create_with_explicit_pk(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
    )
    mm = u.model(gws_root, 'PLAIN')
    f1 = [
        u.feature(mm, {'id': 15, 'str1': 'aa'}),
        u.feature(mm, {'id': 17, 'str1': 'bb'}),
        u.feature(mm, {'id': 19, 'str1': 'cc'}),
    ]
    f2 = mm.create_features(f1, mc)
    assert u.model_feature_atts(f2, 'id') == [15, 17, 19]
    assert u.model_feature_atts(f2, 'str1') == ['aa', 'bb', 'cc']


def test_create_with_auto_pk(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
    )
    mm = u.model(gws_root, 'SERIAL_ID')
    f1 = [
        u.feature(mm, {'str1': '11', 'str2': '21'}),
        u.feature(mm, {'str1': '12', 'str2': '22'}),
        u.feature(mm, {'str1': '13', 'str2': '23'}),
    ]
    f2 = mm.create_features(f1, mc)
    assert u.model_feature_atts(f2, 'id') == [1, 2, 3]
    assert u.model_feature_atts(f2, 'str1') == ['11', '12', '13']
