import gws
import gws.lib.datetimex
import gws.test.util as u
import gws.plugin.postgres.storage_provider


TABLE_NAME = 'test_storage'


@u.fixture(scope='module')
def storage():
    u.pg.exec(f'DROP TABLE IF EXISTS {TABLE_NAME}')
    u.pg.exec(gws.plugin.postgres.storage_provider.TABLE_DDL.format(table_name=TABLE_NAME))

    cfg = f'''
        storage.providers+ {{
            uid "TEST_STORAGE_PROVIDER"
            type "postgres"
            dbUid "GWS_TEST_POSTGRES_PROVIDER"
            tableName "{TABLE_NAME}"
        }}
    '''

    root = u.gws_root(cfg)
    yield u.cast(gws.StorageProvider, root.get('TEST_STORAGE_PROVIDER'))


@u.fixture(autouse=True)
def clear_table():
    u.pg.clear(TABLE_NAME)


def test_list_names_empty(storage: gws.StorageProvider):
    assert storage.list_names('cat1') == []


def test_list_names(storage: gws.StorageProvider):
    u.pg.exec(
        f"INSERT INTO {TABLE_NAME} (category, name, user_uid, data, created, updated) VALUES"
        f" ('cat1', 'b', 'u1', 'd1', to_timestamp(100), to_timestamp(200)),"
        f" ('cat1', 'a', 'u1', 'd2', to_timestamp(101), to_timestamp(201)),"
        f" ('cat1', 'c', 'u1', 'd3', to_timestamp(102), to_timestamp(202)),"
        f" ('cat2', 'x', 'u2', 'd4', to_timestamp(103), to_timestamp(203))"
    )

    assert storage.list_names('cat1') == ['a', 'b', 'c']
    assert storage.list_names('cat2') == ['x']
    assert storage.list_names('cat3') == []


def test_read_missing(storage: gws.StorageProvider):
    assert storage.read('cat1', 'no_such') is None


def test_read(storage: gws.StorageProvider):
    u.pg.exec(
        f"INSERT INTO {TABLE_NAME} (category, name, user_uid, data, created, updated) VALUES"
        f" ('cat1', 'entry1', 'user99', 'hello', to_timestamp(1000000), to_timestamp(2000000))"
    )

    rec = storage.read('cat1', 'entry1')
    assert rec is not None
    assert rec.name == 'entry1'
    assert rec.data == 'hello'
    assert rec.user_uid == 'user99'
    assert isinstance(rec.created, gws.lib.datetimex.dt.datetime)
    assert isinstance(rec.updated, gws.lib.datetimex.dt.datetime)
    assert rec.created == gws.lib.datetimex.from_timestamp(1000000)
    assert rec.updated == gws.lib.datetimex.from_timestamp(2000000)


def test_read_wrong_category(storage: gws.StorageProvider):
    u.pg.exec(
        f"INSERT INTO {TABLE_NAME} (category, name, user_uid, data, created, updated) VALUES"
        f" ('cat1', 'entry1', 'u1', 'hello', to_timestamp(111), to_timestamp(222))"
    )

    assert storage.read('cat2', 'entry1') is None


def test_write_new(storage: gws.StorageProvider):
    storage.write('cat1', 'entry1', 'data1', 'user1')

    rows = u.pg.rows(f"SELECT category, name, user_uid, data FROM {TABLE_NAME} ORDER BY name")
    assert rows == [('cat1', 'entry1', 'user1', 'data1')]


def test_write_update(storage: gws.StorageProvider):
    storage.write('cat1', 'entry1', 'data1', 'user1')
    rec1 = storage.read('cat1', 'entry1')
    created1 = rec1.created

    storage.write('cat1', 'entry1', 'data2', 'user2')

    rows = u.pg.rows(f"SELECT category, name, data FROM {TABLE_NAME} ORDER BY name")
    assert rows == [('cat1', 'entry1', 'data2')]

    rec2 = storage.read('cat1', 'entry1')
    assert rec2.data == 'data2'
    assert rec2.user_uid == 'user2'
    assert rec2.created == created1


def test_write_update_preserves_created(storage: gws.StorageProvider):
    storage.write('cat1', 'entry1', 'data1', 'user1')
    rec1 = storage.read('cat1', 'entry1')

    storage.write('cat1', 'entry1', 'data2', 'user1')
    rec2 = storage.read('cat1', 'entry1')

    assert rec2.created == rec1.created
    assert rec2.updated >= rec1.updated


def test_write_multiple_categories(storage: gws.StorageProvider):
    storage.write('cat1', 'entry1', 'data_a', 'user1')
    storage.write('cat2', 'entry1', 'data_b', 'user2')

    rec_a = storage.read('cat1', 'entry1')
    rec_b = storage.read('cat2', 'entry1')

    assert rec_a.data == 'data_a'
    assert rec_b.data == 'data_b'


def test_delete(storage: gws.StorageProvider):
    storage.write('cat1', 'entry1', 'data1', 'user1')
    storage.write('cat1', 'entry2', 'data2', 'user1')

    storage.delete('cat1', 'entry1')

    assert storage.read('cat1', 'entry1') is None
    assert storage.read('cat1', 'entry2') is not None
    assert storage.list_names('cat1') == ['entry2']


def test_delete_nonexistent(storage: gws.StorageProvider):
    # deleting a non-existent entry should not raise
    storage.delete('cat1', 'no_such')


def test_delete_only_matching_category(storage: gws.StorageProvider):
    storage.write('cat1', 'entry1', 'data1', 'user1')
    storage.write('cat2', 'entry1', 'data2', 'user2')

    storage.delete('cat1', 'entry1')

    assert storage.read('cat1', 'entry1') is None
    assert storage.read('cat2', 'entry1') is not None
