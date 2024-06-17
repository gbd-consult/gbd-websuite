import gws.base.database
import gws.lib.sa as sa
import gws.test.util as u


@u.fixture(scope='module')
def root():
    u.pg.create('plain', {'id': 'int primary key', 'a': 'text', 'b': 'text', 'c': 'text'})
    cfg = '''
        models+ { 
            uid "PLAIN" type "postgres" tableName "plain" 
        }
        models+ { 
            uid "SERIAL_ID" type "postgres" tableName "serial_id"
        }
    '''

    yield u.gws_root(cfg)


##

def test_error(root: gws.Root):
    db = u.get_db(root)

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


def test_error_rollback(root: gws.Root):
    db = u.get_db(root)

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


def test_error_no_rollback(root: gws.Root):
    db = u.get_db(root)

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


def test_nested_connections(root: gws.Root):
    db = u.get_db(root)
    db_name = u.option('service.postgres.database')

    def _num_conn(c):
        return (
            c.execute(
                sa.text(f"SELECT numbackends FROM pg_stat_database WHERE datname='{db_name}' "))
        ).scalar_one()

    with db.connect() as c1:
        n1 = _num_conn(c1)
        with db.connect() as c2:
            n2 = _num_conn(c2)
            with db.connect() as c3:
                n3 = _num_conn(c3)

    assert n1 == n2 == n3


def test_nested_connection_opens_once(root: gws.Root):
    db = u.get_db(root)

    log = []

    class MockConn:
        def __init__(self):
            log.append('OPEN')

        def execute(self, s):
            log.append(f'EXEC {s}')

        def close(self):
            log.append('CLOSE')

    with u.monkey_patch() as mp:
        mp.setattr(sa.Engine, 'connect', lambda *args: MockConn())

        with db.connect() as conn:
            conn.execute('1')
            with db.connect() as conn:
                conn.execute('2')
                with db.connect() as conn:
                    conn.execute('3')

    assert log == ['OPEN', 'EXEC 1', 'EXEC 2', 'EXEC 3', 'CLOSE']
