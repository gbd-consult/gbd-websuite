import gws
import gws.test.util as u


@u.fixture(scope='module')
def root():
    cfg = '''
        auth {
            providers+ {
                type 'provider1'
                allowedMethods ['method1']
            }
            methods+ { type 'method1' }
            methods+ { type 'method2' }
            
            session.lifeTime 2
        }
    '''

    yield u.gws_root(cfg)


##


def test_authenticate(root: gws.Root):
    am = root.app.authMgr
    u0 = u.mocks.USERS[0]
    u1 = am.authenticate(am.methods[0], gws.Data(login=u0['login'], password=u0['password']))
    assert u1.displayName == u0['displayName']


def test_authenticate_with_bad_login_fails(root: gws.Root):
    am = root.app.authMgr
    u1 = am.authenticate(am.methods[0], gws.Data(login='BAD'))
    assert u1 is None


def test_authenticate_with_wrong_method_fails(root: gws.Root):
    am = root.app.authMgr
    u0 = u.mocks.USERS[0]
    u1 = am.authenticate(am.methods[1], gws.Data(login=u0['login'], password=u0['password']))
    assert u1 is None


def test_get_user(root: gws.Root):
    am = root.app.authMgr
    u0 = u.mocks.USERS[0]
    u1 = am.authenticate(am.methods[0], gws.Data(login=u0['login'], password=u0['password']))
    assert am.get_user(u1.uid).displayName == u0['displayName']
