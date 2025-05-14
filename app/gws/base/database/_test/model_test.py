import os
import gws.config
import gws.base.feature
import gws.lib.sa as sa
import gws.test.util as u


##

_weird_names = [
    'Weird.Schema.. ""äöü"" !',
    'Weird.Table..  ""äöü"" !',
    'Weird.Column.. ""äöü"" !',
]

@u.fixture(scope='module')
def root():
    u.pg.create('plain', {'id': 'int primary key', 'a': 'text'})
    u.pg.create('auto_pk', {'id': 'serial primary key', 'a': 'text'})
    u.pg.create('no_pk', {'id': 'int', 'a': 'text'})

    ws, wt, wc = _weird_names
    ddl = f'''
        drop schema if exists "{ws}" cascade;
        create schema "{ws}";
        create table "{ws}"."{wt}" (
            id integer primary key,
            "{wc}" text
        )
    '''

    conn = u.pg.connect()
    for d in ddl.split(';'):
        conn.execute(sa.text(d))
        conn.commit()

    cfg = f'''
        models+ {{ 
            uid "PLAIN" type "postgres" tableName "plain" 
        }}
        models+ {{
            uid "AUTO_PK" type "postgres" tableName "auto_pk"
        }}
        models+ {{
            uid "NO_PK" type "postgres" tableName "no_pk"
        }}
        models+ {{
            uid "WEIRD" type "postgres" tableName ' "{ws}"."{wt}" '
        }}
    '''

    yield u.gws_root(cfg)


##

def test_get_features(root: gws.Root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        op=gws.ModelOperation.read,
    )
    mo = u.cast(gws.Model, root.get('PLAIN'))

    u.pg.insert('plain', [
        dict(id=1, a='11'),
        dict(id=2, a='22'),
        dict(id=3, a='33'),
        dict(id=4, a='44'),
    ])

    fs = mo.get_features([2, 3], mc)
    assert [f.get('a') for f in fs] == ['22', '33']


##

def test_create_with_explicit_pk(root: gws.Root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
    )
    mo = u.cast(gws.Model, root.get('PLAIN'))

    u.pg.clear('plain')

    mo.create_feature(u.feature(mo, id=15, a='aa'), mc)
    mo.create_feature(u.feature(mo, id=16, a='bb'), mc)
    mo.create_feature(u.feature(mo, id=17, a='cc'), mc)

    assert u.pg.content('select id, a from plain') == [
        (15, 'aa'),
        (16, 'bb'),
        (17, 'cc'),
    ]


def test_create_with_auto_pk(root: gws.Root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
    )
    mo = u.cast(gws.Model, root.get('AUTO_PK'))

    u.pg.clear('auto_pk')

    mo.create_feature(u.feature(mo, a='aa'), mc)
    mo.create_feature(u.feature(mo, a='bb'), mc)
    mo.create_feature(u.feature(mo, a='cc'), mc)

    assert u.pg.content('auto_pk') == [
        (1, 'aa'),
        (2, 'bb'),
        (3, 'cc'),
    ]

def test_create_no_pk(root: gws.Root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
    )
    mo = u.cast(gws.Model, root.get('NO_PK'))

    u.pg.clear('no_pk')

    mo.create_feature(u.feature(mo, id=11, a='aa'), mc)
    mo.create_feature(u.feature(mo, id=22, a='bb'), mc)
    mo.create_feature(u.feature(mo, id=33, a='cc'), mc)

    assert u.pg.content('no_pk') == [
        (11, 'aa'),
        (22, 'bb'),
        (33, 'cc'),
    ]

def test_weird_names(root: gws.Root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
    )
    mo = u.cast(gws.Model, root.get('WEIRD'))

    ws, wt, wc = _weird_names
    wc_noquot = wc.replace('""', '"')

    u.pg.clear(f'"{ws}"."{wt}"')

    mo.create_feature(u.feature_from_dict(mo, {'id': 15, wc_noquot: 'aa'}), mc)
    mo.create_feature(u.feature_from_dict(mo, {'id': 16, wc_noquot: 'bb'}), mc)
    mo.create_feature(u.feature_from_dict(mo, {'id': 17, wc_noquot: 'cc'}), mc)

    assert u.pg.content(f'select id, "{wc}" from "{ws}"."{wt}"') == [
        (15, 'aa'),
        (16, 'bb'),
        (17, 'cc'),
    ]
