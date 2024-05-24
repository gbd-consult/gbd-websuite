import os
import gws.config
import gws.base.database
import gws.lib.sa as sa
import gws.test.util as u


##


@u.fixture(scope='module')
def gws_root():
    u.pg_create('plain', {'id': 'int primary key', 'a': 'text', 'b': 'text', 'c': 'text'})
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

def test_error(gws_root):
    db = gws_root.get('GWS_TEST_POSTGRES_PROVIDER')

    err = ''
    with db.connect() as conn:
        try:
            conn.execute('not an object')
        except sa.Error as e:
            err = str(e)
    assert 'Not an executable object' in err

    err = ''
    with db.connect() as conn:
        try:
            conn.execute(sa.text('select * from not_a_table'))
        except sa.Error as e:
            err = str(e)
    assert 'UndefinedTable' in err


def test_error_rollback(gws_root):
    db = gws_root.get('GWS_TEST_POSTGRES_PROVIDER')

    err = ''
    with db.connect() as conn:
        try:
            conn.execute(sa.text('select * from not_a_table'))
        except sa.Error as e:
            err = str(e)
            conn.rollback()
        try:
            conn.execute(sa.text('select 1'))
        except sa.Error as e:
            err = 'FAILED'
    assert 'UndefinedTable' in err


def test_error_no_rollback(gws_root):
    db = gws_root.get('GWS_TEST_POSTGRES_PROVIDER')

    err = ''
    with db.connect() as conn:
        try:
            conn.execute(sa.text('select * from not_a_table'))
        except sa.Error as e:
            err = str(e)
            # no rollback
        try:
            conn.execute(sa.text('select 1'))
        except sa.Error as e:
            # ERROR:  current transaction is aborted...
            err = 'FAILED'
    assert 'FAILED' in err
