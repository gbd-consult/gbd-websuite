import time
import gws
import gws.lib.osx
import gws.test.util as u
from gws.base.auth import method, provider, user


class FooMethod(method.Object):
    pass


class BarMethod(method.Object):
    pass


class MockProvider(provider.Object):
    def authenticate(self, method, credentials):
        if credentials.get('login') == 'OK':
            return self.get_user('user1')

    def get_user(self, local_uid):
        if local_uid == 'user1':
            return user.create(user.AuthorizedUser, self, 'user1', ['role1'], {'displayName': 'USER_1'})


@u.fixture(scope='module')
def auth():

    defaults = {
        'providers': [
            {
                'type': 'mock',
                'allowedMethods': ['foo'],
            }
        ],
        'methods': [
            {
                'type': 'foo'
            },
            {
                'type': 'bar'
            }
        ],
        'sessionLifeTime': 2,
    }

    yield u.gws_configure('auth defaults').application.auth


##


def test_authenticate(auth):
    user = auth.authenticate(auth.methods[0], gws.Data(login='OK'))
    assert user.displayName == 'USER_1'


def test_authenticate_with_bad_login_fails(auth):
    user = auth.authenticate(auth.methods[0], gws.Data(login='BAD'))
    assert user is None


def test_authenticate_with_wrong_method_fails(auth):
    user = auth.authenticate(auth.methods[1], gws.Data(login='OK'))
    assert user is None


def test_get_user(auth):
    user = auth.authenticate(auth.methods[0], gws.Data(login='OK'))
    assert vars(auth.get_user(user.uid)) == vars(user)


def test_create_stored_session(auth):
    user = auth.authenticate(auth.methods[0], gws.Data(login='OK'))
    sess1 = auth.new_stored_session('test', auth.methods[0], user)
    sess2 = auth.find_stored_session(sess1.uid)
    assert sess1.user.uid == sess2.user.uid
    assert sess1.method.uid == sess2.method.uid


def test_update_stored_session(auth):
    user = auth.authenticate(auth.methods[0], gws.Data(login='OK'))
    sess1 = auth.new_stored_session('test', auth.methods[0], user)
    sess1.set('foo', 'bar')
    auth.save_stored_session(sess1)
    sess2 = auth.find_stored_session(sess1.uid)
    assert sess2.get('foo') == 'bar'


def test_destroy_stored_session(auth):
    user = auth.authenticate(auth.methods[0], gws.Data(login='OK'))
    sess1 = auth.new_stored_session('test', auth.methods[0], user)
    auth.destroy_stored_session(sess1)
    sess2 = auth.find_stored_session(sess1.uid)
    assert sess2 is None


def test_stored_session_expiration(auth):
    user = auth.authenticate(auth.methods[0], gws.Data(login='OK'))
    dead = auth.new_stored_session('test', auth.methods[0], user)
    live = auth.new_stored_session('test', auth.methods[0], user)

    time.sleep(1)
    auth.save_stored_session(live)
    time.sleep(1)
    auth.save_stored_session(live)
    time.sleep(1)

    dead2 = auth.find_stored_session(dead.uid)
    live2 = auth.find_stored_session(live.uid)

    assert dead2 is None
    assert live2.uid == live.uid
