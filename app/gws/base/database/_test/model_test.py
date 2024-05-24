import os
import gws.config
import gws.base.feature
import gws.lib.sa as sa
import gws.test.util as u


##


@u.fixture(scope='module')
def gws_root():
    u.pg_create('plain', {'id': 'int primary key', 'a': 'text', 'b': 'text', 'c': 'text'})
    u.pg_create('serial_id', {'id': 'serial primary key', 'a': 'text'})

    cfg = '''
        models+ { 
            uid "PLAIN" type "postgres" tableName "plain" 
        }
        models+ { 
            uid "SERIAL_ID" type "postgres" tableName "serial_id"
        }
    '''

    yield u.gws_configure(cfg)


##

def test_get_features(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        op=gws.ModelOperation.read,
    )
    mo = u.model(gws_root, 'PLAIN')

    u.pg_insert('plain', [
        dict(id=1, a='11'),
        dict(id=2, a='22'),
        dict(id=3, a='33'),
        dict(id=4, a='44'),
    ])

    fs = mo.get_features([2, 3], mc)
    assert [f.get('a') for f in fs] == ['22', '33']


##

def test_create_with_explicit_pk(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
    )
    mo = u.model(gws_root, 'PLAIN')

    u.pg_clear('plain')

    mo.create_feature(u.feature(mo, id=15, a='aa'), mc)
    mo.create_feature(u.feature(mo, id=16, a='bb'), mc)
    mo.create_feature(u.feature(mo, id=17, a='cc'), mc)

    assert u.pg_content('select id, a from plain') == [
        (15, 'aa'),
        (16, 'bb'),
        (17, 'cc'),
    ]


def test_create_with_auto_pk(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
    )
    mo = u.model(gws_root, 'SERIAL_ID')

    u.pg_clear('serial_id')

    mo.create_feature(u.feature(mo, a='aa'), mc)
    mo.create_feature(u.feature(mo, a='bb'), mc)
    mo.create_feature(u.feature(mo, a='cc'), mc)

    assert u.pg_content('serial_id') == [
        (1, 'aa'),
        (2, 'bb'),
        (3, 'cc'),
    ]
