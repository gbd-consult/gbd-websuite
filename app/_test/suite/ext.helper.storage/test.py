import gws.config.loader
import gws.ext.helper.storage
import _test.util as u

import gws.types as t

d1 = {'name': 'Name1', 'num': 11, 'prop': 'Val1'}
d2 = {'name': 'Name2', 'num': 22, 'prop': 'Val2'}
d3 = {'name': 'Name3', 'num': 33, 'prop': 'Val3'}

DB_PATH = "/gws-var/test_storage.sqlite"
PASSWORD = '123'


def data(category, n):
    return {'name': f'_{category}_{n}', 'num': n * 10}


storage_obj = t.cast(
    gws.ext.helper.storage.Object,
    gws.config.loader.load().find_first('gws.ext.helper.storage'))


def populate_storage():
    storage_obj.reset()

    auth = u.cmd('authLogin', {'username': 'ddd-power', 'password': PASSWORD}).cookies

    for cat in 'cat_read', 'cat_write', 'cat_all_other_read':
        u.cmd('storageWrite', {'entry': {'category': cat, 'name': '1'}, 'data': data(cat, 1)}, cookies=auth)
        u.cmd('storageWrite', {'entry': {'category': cat, 'name': '2'}, 'data': data(cat, 2)}, cookies=auth)
        u.cmd('storageWrite', {'entry': {'category': cat, 'name': '3'}, 'data': data(cat, 3)}, cookies=auth)


def test_dir():
    populate_storage()

    # normal dir

    auth = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_all_other_read'
    r = u.cmd('storageDir', {'category': cat}, cookies=auth)
    assert r.json() == {
        'entries': [
            {'category': cat, 'name': '1'},
            {'category': cat, 'name': '2'},
            {'category': cat, 'name': '3'}
        ],
        'readable': True,
        'writable': True
    }

    # read only

    auth = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_read'
    r = u.cmd('storageDir', {'category': cat}, cookies=auth)
    assert r.json() == {
        'entries': [
            {'category': cat, 'name': '1'},
            {'category': cat, 'name': '2'},
            {'category': cat, 'name': '3'}
        ],
        'readable': True,
        'writable': False
    }

    # write only

    auth = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_write'
    r = u.cmd('storageDir', {'category': cat}, cookies=auth)
    assert r.json() == {
        'entries': [],
        'readable': False,
        'writable': False
    }

    # no access

    auth = u.cmd('authLogin', {'username': 'ccc-no-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_read'
    r = u.cmd('storageDir', {'category': cat}, cookies=auth)
    assert r.json() == {
        'entries': [],
        'readable': False,
        'writable': False,
    }


def test_read():
    populate_storage()

    # read/all

    auth = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_all_other_read'
    r = u.cmd('storageRead', {'entry': {'category': cat, 'name': '1'}}, cookies=auth)
    assert r.json() == {
        'entry': {'category': cat, 'name': '1'},
        'data': data(cat, 1)
    }

    # read/all

    auth = u.cmd('authLogin', {'username': 'ccc-no-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_all_other_read'
    r = u.cmd('storageRead', {'entry': {'category': cat, 'name': '1'}}, cookies=auth)
    assert r.json() == {
        'entry': {'category': cat, 'name': '1'},
        'data': data(cat, 1)
    }

    # read/read

    auth = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_read'
    r = u.cmd('storageRead', {'entry': {'category': cat, 'name': '1'}}, cookies=auth)
    assert r.json() == {
        'entry': {'category': cat, 'name': '1'},
        'data': data(cat, 1)
    }

    # write only

    auth = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_write'
    r = u.cmd('storageRead', {'entry': {'category': cat, 'name': 'NOT_FOUND'}}, cookies=auth)
    assert r.status_code == 403

    # not found

    auth = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_read'
    r = u.cmd('storageRead', {'entry': {'category': cat, 'name': 'NOT_FOUND'}}, cookies=auth)
    assert r.status_code == 404

    # no access

    auth = u.cmd('authLogin', {'username': 'ccc-no-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_read'
    r = u.cmd('storageRead', {'entry': {'category': cat, 'name': '1'}}, cookies=auth)
    assert r.status_code == 403


def test_write():
    populate_storage()

    # write new and read back

    auth = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_all_other_read'
    r = u.cmd('storageWrite', {'entry': {'category': cat, 'name': '100'}, 'data': data(cat, 100)}, cookies=auth)
    assert r.json() == {
        'entry': {'category': cat, 'name': '100'},
    }

    auth = u.cmd('authLogin', {'username': 'ccc-no-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_all_other_read'
    r = u.cmd('storageRead', {'entry': {'category': cat, 'name': '100'}}, cookies=auth)
    assert r.json() == {
        'entry': {'category': cat, 'name': '100'},
        'data': data(cat, 100),
    }

    # overwrite and read back

    new_data = {'s': 'NEWDATA'}

    auth = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_all_other_read'
    r = u.cmd('storageWrite', {'entry': {'category': cat, 'name': '2'}, 'data': new_data}, cookies=auth)
    assert r.json() == {
        'entry': {'category': cat, 'name': '2'},
    }

    auth = u.cmd('authLogin', {'username': 'ccc-no-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_all_other_read'
    r = u.cmd('storageRead', {'entry': {'category': cat, 'name': '2'}}, cookies=auth)
    assert r.json() == {
        'entry': {'category': cat, 'name': '2'},
        'data': new_data,
    }

    # read only

    auth = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_read'
    r = u.cmd('storageWrite', {'entry': {'category': cat, 'name': '100'}, 'data': data(cat, 100)}, cookies=auth)
    assert r.status_code == 403

    # no access

    auth = u.cmd('authLogin', {'username': 'ccc-no-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_all_other_read'
    r = u.cmd('storageWrite', {'entry': {'category': cat, 'name': '100'}, 'data': data(cat, 100)}, cookies=auth)
    assert r.status_code == 403


def test_delete():
    populate_storage()

    # delete

    auth = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_all_other_read'
    r = u.cmd('storageDelete', {'entry': {'category': cat, 'name': '2'}}, cookies=auth)
    r = u.cmd('storageDir', {'category': cat}, cookies=auth)
    assert r.json() == {
        'entries': [
            {'category': cat, 'name': '1'},
            {'category': cat, 'name': '3'}
        ],
        'readable': True,
        'writable': True
    }

    # read only

    auth = u.cmd('authLogin', {'username': 'aaa-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_read'
    r = u.cmd('storageDelete', {'entry': {'category': cat, 'name': '2'}}, cookies=auth)
    assert r.status_code == 403

    # no access

    auth = u.cmd('authLogin', {'username': 'ccc-no-testrole', 'password': PASSWORD}).cookies
    cat = 'cat_all_other_read'
    r = u.cmd('storageDelete', {'entry': {'category': cat, 'name': '2'}}, cookies=auth)
    assert r.status_code == 403
