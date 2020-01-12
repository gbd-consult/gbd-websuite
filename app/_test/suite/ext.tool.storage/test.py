import gws.config.loader
import _test.util as u

d1 = {'name': 'Name1', 'num': 11, 'prop': 'Val1'}
d2 = {'name': 'Name2', 'num': 22, 'prop': 'Val2'}
d3 = {'name': 'Name3', 'num': 33, 'prop': 'Val3'}

DB_PATH = "/gws-var/test_storage.sqlite"


def test_read_write():
    gws.config.loader.load().find_first('gws.ext.tool.storage').reset()

    r = u.cmd('storageWrite', {'entry': {'category': 'abc', 'name': '1'}, 'data': d1})
    assert r.json() == {'entry': {'category': 'abc', 'name': '1'}}

    r = u.cmd('storageWrite', {'entry': {'category': 'abc', 'name': '2'}, 'data': d2})
    assert r.json() == {'entry': {'category': 'abc', 'name': '2'}}

    r = u.cmd('storageWrite', {'entry': {'category': 'def', 'name': '3'}, 'data': d3})
    assert r.json() == {'entry': {'category': 'def', 'name': '3'}}

    r = u.cmd('storageDir', {'category': 'abc', })

    assert r.json() == {'entries': [
        {'category': 'abc', 'name': '1'},
        {'category': 'abc', 'name': '2'},
    ]}

    r = u.cmd('storageDir', {'category': 'not_found', })
    assert r.json() == {'entries': []}

    r = u.cmd('storageRead', {'entry': {'category': 'abc', 'name': '1'}, })
    assert r.json() == {'entry': {'category': 'abc', 'name': '1'}, 'data': d1}

    r = u.cmd('storageRead', {'entry': {'category': 'abc', 'name': '2'}, })
    assert r.json() == {'entry': {'category': 'abc', 'name': '2'}, 'data': d2}

    r = u.cmd('storageRead', {'entry': {'category': 'def', 'name': '3'}, })
    assert r.json() == {'entry': {'category': 'def', 'name': '3'}, 'data': d3}

    r = u.cmd('storageRead', {'entry': {'category': 'def', 'name': 'not_found'}, })
    assert r.status_code == 404

    r = u.cmd('storageRead', {'entry': {'category': 'not_found', 'name': '1'}, })
    assert r.status_code == 404
