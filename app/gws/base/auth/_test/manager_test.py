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
            methods+ { type 'mockAuthMethod2' }
        }
    '''

    yield u.gws_root(cfg)


##


def test_authenticate(root: gws.Root):
    am = root.app.authMgr
    u.mock.add_user('me', 'foo', displayName='123')
    usr = am.authenticate(am.methods[0], gws.Data(username='me', password='foo'))
    assert usr.displayName == '123'


def test_authenticate_with_bad_login_fails(root: gws.Root):
    am = root.app.authMgr
    u.mock.add_user('me', 'foo')
    usr = am.authenticate(am.methods[0], gws.Data(username='BAD', password='foo'))
    assert usr is None


def test_authenticate_with_wrong_method_fails(root: gws.Root):
    am = root.app.authMgr
    u.mock.add_user('me', 'foo')
    usr = am.authenticate(am.methods[1], gws.Data(username='me', password='foo'))
    assert usr is None


def test_get_user(root: gws.Root):
    am = root.app.authMgr
    u.mock.add_user('me', 'foo', displayName='890')
    usr = am.authenticate(am.methods[0], gws.Data(username='me', password='foo'))
    assert am.get_user(usr.uid).displayName == '890'
