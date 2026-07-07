import datetime

import gws
import gws.lib.datetimex
import gws.test.util as u
import gws.plugin.storage_provider.sqlite


@u.fixture()
def provider(tmp_path):
    db_path = str(tmp_path / 'storage.sqlite')
    cfg = f'''
        storage.providers+ {{
            uid "TEST_SQLITE_STORAGE"
            type "sqlite"
            path "{db_path}"
        }}
    '''
    root = u.gws_root(cfg)
    provider = u.cast(gws.StorageProvider, root.get('TEST_SQLITE_STORAGE'))
    provider._db()  # initialise table
    yield provider


def _db(storage):
    return storage._db()


def test_list_names_empty(provider):
    assert provider.list_names('cat1') == []


def test_list_names(provider):
    db = _db(provider)
    db.insert('storage', {'category': 'cat1', 'name': 'b', 'user_uid': 'u1', 'data': 'd1', 'created': 100, 'updated': 200})
    db.insert('storage', {'category': 'cat1', 'name': 'a', 'user_uid': 'u1', 'data': 'd2', 'created': 101, 'updated': 201})
    db.insert('storage', {'category': 'cat1', 'name': 'c', 'user_uid': 'u1', 'data': 'd3', 'created': 102, 'updated': 202})
    db.insert('storage', {'category': 'cat2', 'name': 'x', 'user_uid': 'u2', 'data': 'd4', 'created': 103, 'updated': 203})

    assert provider.list_names('cat1') == ['a', 'b', 'c']
    assert provider.list_names('cat2') == ['x']
    assert provider.list_names('cat3') == []


def test_read_missing(provider):
    assert provider.read('cat1', 'no_such') is None


def test_read(provider):
    ts_created = 1_000_000
    ts_updated = 2_000_000
    _db(provider).insert('storage', {
        'category': 'cat1', 'name': 'entry1', 'user_uid': 'user99',
        'data': 'hello', 'created': ts_created, 'updated': ts_updated,
    })

    rec = provider.read('cat1', 'entry1')
    assert rec is not None
    assert rec.name == 'entry1'
    assert rec.data == 'hello'
    assert rec.user_uid == 'user99'
    assert isinstance(rec.created, datetime.datetime)
    assert isinstance(rec.updated, datetime.datetime)
    assert rec.created == gws.lib.datetimex.from_timestamp(ts_created)
    assert rec.updated == gws.lib.datetimex.from_timestamp(ts_updated)


def test_read_wrong_category(provider):
    _db(provider).insert('storage', {'category': 'cat1', 'name': 'entry1', 'user_uid': 'u1', 'data': 'hello', 'created': 100, 'updated': 200})
    assert provider.read('cat2', 'entry1') is None


def test_write_new(provider):
    provider.write('cat1', 'entry1', 'data1', 'user1')

    rows = _db(provider).select('SELECT category, name, user_uid, data FROM storage ORDER BY name')
    assert len(rows) == 1
    assert rows[0]['name'] == 'entry1'
    assert rows[0]['data'] == 'data1'
    assert rows[0]['user_uid'] == 'user1'


def test_write_new_timestamps_are_datetimes(provider):
    provider.write('cat1', 'entry1', 'data1', 'user1')

    rec = provider.read('cat1', 'entry1')
    assert isinstance(rec.created, datetime.datetime)
    assert isinstance(rec.updated, datetime.datetime)


def test_write_update(provider):
    provider.write('cat1', 'entry1', 'data1', 'user1')
    rec1 = provider.read('cat1', 'entry1')

    provider.write('cat1', 'entry1', 'data2', 'user2')
    rec2 = provider.read('cat1', 'entry1')

    assert rec2.data == 'data2'
    assert rec2.user_uid == 'user2'
    assert rec2.created == rec1.created


def test_write_update_preserves_created(provider):
    provider.write('cat1', 'entry1', 'data1', 'user1')
    rec1 = provider.read('cat1', 'entry1')

    provider.write('cat1', 'entry1', 'data2', 'user1')
    rec2 = provider.read('cat1', 'entry1')

    assert rec2.created == rec1.created
    assert rec2.updated >= rec1.updated


def test_write_multiple_categories(provider):
    provider.write('cat1', 'entry1', 'data_a', 'user1')
    provider.write('cat2', 'entry1', 'data_b', 'user2')

    rec_a = provider.read('cat1', 'entry1')
    rec_b = provider.read('cat2', 'entry1')

    assert rec_a.data == 'data_a'
    assert rec_b.data == 'data_b'


def test_delete(provider):
    provider.write('cat1', 'entry1', 'data1', 'user1')
    provider.write('cat1', 'entry2', 'data2', 'user1')

    provider.delete('cat1', 'entry1')

    assert provider.read('cat1', 'entry1') is None
    assert provider.read('cat1', 'entry2') is not None
    assert provider.list_names('cat1') == ['entry2']


def test_delete_nonexistent(provider):
    # deleting a non-existent entry should not raise
    provider.delete('cat1', 'no_such')


def test_delete_only_matching_category(provider):
    provider.write('cat1', 'entry1', 'data1', 'user1')
    provider.write('cat2', 'entry1', 'data2', 'user2')

    provider.delete('cat1', 'entry1')

    assert provider.read('cat1', 'entry1') is None
    assert provider.read('cat2', 'entry1') is not None
