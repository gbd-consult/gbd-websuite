import multiprocessing

import gws
import gws.lib.osx as osx
import gws.test.util as u

DB_PATH = u.path_in_base_dir('sess')


def root():
    cfg = f'''
        auth {{
            providers+ {{
                type 'mockAuthProvider1'
                allowedMethods ['mockAuthMethod1']
            }}
            methods+ {{ 
                type 'mockAuthMethod1' 
            }}
            session {{
                path {DB_PATH!r}
                type "sqlite"
                lifeTime 2
            }}
        }}
    '''

    return u.gws_root(cfg)


##

def _prepare() -> tuple[gws.AuthManager, gws.AuthSessionManager, gws.User]:
    am = root().app.authMgr
    sm = am.sessionMgr
    u.mock.add_user('me', 'foo')
    usr = am.authenticate(am.methods[0], gws.Data(username='me', password='foo'))
    return am, sm, usr


def test_db_recreated_if_deleted():
    osx.unlink(DB_PATH)

    am, sm, usr = _prepare()
    s1 = sm.create(am.methods[0], usr, {'foo': 'bar'})
    assert sm.get_valid(s1.uid).get('foo') == 'bar'

    osx.unlink(DB_PATH)

    # should not raise
    assert sm.get_valid(s1.uid) is None

    # 'save' has no effect, the session is gone
    s1.set('foo', 'bar2')
    sm.save(s1)
    assert sm.get_valid(s1.uid) is None

    # 'create' is fine
    s2 = sm.create(am.methods[0], usr, {'foo': 'bar3'})
    assert sm.get_valid(s2.uid).get('foo') == 'bar3'


def test_create_session():
    osx.unlink(DB_PATH)
    am, sm, usr = _prepare()

    s1 = sm.create(am.methods[0], usr)
    s2 = sm.get_valid(s1.uid)

    assert s1.user.uid == s2.user.uid
    assert s1.method.uid == s2.method.uid


def test_update_session():
    osx.unlink(DB_PATH)
    am, sm, usr = _prepare()

    s1 = sm.create(am.methods[0], usr)
    s1.set('foo', 'bar')
    u1 = s1.updated
    gws.u.sleep(1)
    sm.save(s1)
    s2 = sm.get_valid(s1.uid)
    u2 = s2.updated
    assert s2.get('foo') == 'bar'
    assert u2 > u1


def test_delete_session():
    osx.unlink(DB_PATH)
    am, sm, usr = _prepare()

    s1 = sm.create(am.methods[0], usr)
    sm.delete(s1)
    s2 = sm.get(s1.uid)
    assert s2 is None


def test_session_expiration():
    osx.unlink(DB_PATH)
    am, sm, usr = _prepare()

    dead = sm.create(am.methods[0], usr)
    live = sm.create(am.methods[0], usr)

    gws.u.sleep(1)

    sm.touch(live)

    gws.u.sleep(1)

    sm.touch(live)

    gws.u.sleep(1)

    dead2 = sm.get_valid(dead.uid)
    live2 = sm.get_valid(live.uid)

    assert dead2 is None
    assert live2.uid == live.uid


def _session_mp_worker(n, num_loops):
    am, sm, usr = _prepare()
    s1 = sm.create(am.methods[0], usr)

    for k in range(num_loops):
        # gws.log.debug(f'_session_mp_worker: sid={s1.uid} {n}:{k}')
        s2 = sm.get(s1.uid)
        s2.set('foo', f'{n}:{k}')
        sm.save(s2)


def test_concurrency():
    num_processes = 100
    num_loops = 50

    osx.unlink(DB_PATH)

    ps = []

    for n in range(num_processes):
        p = multiprocessing.Process(target=_session_mp_worker, args=[n, num_loops])
        ps.append(p)
        p.start()

    for p in ps:
        p.join()

    # ensure everything is written

    am, sm, usr = _prepare()
    all_sess = sm.get_all()
    assert len(all_sess) == num_processes

    v1 = set(f'{n}:{num_loops - 1}' for n in range(num_processes))
    v2 = set(s.get('foo') for s in all_sess)
    assert v2 == v1
