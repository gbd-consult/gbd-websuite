import gws
import gws.test.util as u


@u.fixture(scope='module')
def root():
    cfg = '''
        auth {
            providers+ {
                type 'mockAuthProvider1'
                allowedMethods ['mockAuthMethod1']
            }
            methods+ { type 'mockAuthMethod1' }
            session {
                type "sqlite"
                lifeTime 2
            }
        }
    '''

    yield u.gws_root(cfg)


##

def _auth(root) -> tuple[gws.AuthManager, gws.AuthSessionManager, gws.User]:
    am = root.app.authMgr
    se = am.sessionMgr
    u.mock.add_user('me', 'foo')
    usr = am.authenticate(am.methods[0], gws.Data(username='me', password='foo'))
    return am, se, usr


def test_create_session(root: gws.Root):
    am, se, usr = _auth(root)

    sess1 = se.create(am.methods[0], usr)
    sess2 = se.get_valid(sess1.uid)

    assert sess1.user.uid == sess2.user.uid
    assert sess1.method.uid == sess2.method.uid


def test_update_session(root: gws.Root):
    am, se, usr = _auth(root)

    sess1 = se.create(am.methods[0], usr)
    sess1.set('foo', 'bar')
    se.save(sess1)
    sess2 = se.get_valid(sess1.uid)
    assert sess2.get('foo') == 'bar'


def test_delete_session(root: gws.Root):
    am, se, usr = _auth(root)

    sess1 = se.create(am.methods[0], usr)
    se.delete(sess1)
    sess2 = se.get(sess1.uid)
    assert sess2 is None


def test_session_expiration(root: gws.Root):
    am, se, usr = _auth(root)

    dead = se.create(am.methods[0], usr)
    live = se.create(am.methods[0], usr)

    gws.u.sleep(1)

    se.touch(live)

    gws.u.sleep(1)

    se.touch(live)

    gws.u.sleep(1)

    dead2 = se.get_valid(dead.uid)
    live2 = se.get_valid(live.uid)

    assert dead2 is None
    assert live2.uid == live.uid
