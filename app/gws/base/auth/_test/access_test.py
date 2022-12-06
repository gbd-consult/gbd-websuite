import gws
import gws.types as t

from gws.base.auth import method, provider, user

mock_provider = t.cast(gws.IAuthProvider, gws.Data(uid='mock_provider'))


def test_access_direct():
    u = user.from_args(user.AuthorizedUser, mock_provider, roles=['a', 'b'])

    obj = gws.Data(access=gws.parse_acl('allow a'))
    assert u.can_use(obj) is True

    obj = gws.Data(access=gws.parse_acl('deny a'))
    assert u.can_use(obj) is False

    obj = gws.Data(access=gws.parse_acl('deny x, allow b'))
    assert u.can_use(obj) is True

    obj = gws.Data(access=gws.parse_acl('deny x, deny b'))
    assert u.can_use(obj) is False

    obj = gws.Data(access=gws.parse_acl('deny x'))
    assert u.can_use(obj) is False


def test_access_context():
    u = user.from_args(user.AuthorizedUser, mock_provider, roles=['a', 'b'])

    x = gws.Data(access=gws.parse_acl('allow x'))
    y = gws.Data(access=gws.parse_acl('allow y'))
    z = gws.Data(access=gws.parse_acl('allow z'))
    a = gws.Data(access=gws.parse_acl('allow a'))

    assert u.can_use(x) is False
    assert u.can_use(x, y, z) is False
    assert u.can_use(x, y, z, a) is True


def test_access_context_and_parent():
    u = user.from_args(user.AuthorizedUser, mock_provider, roles=['a', 'b'])

    x = gws.Data(access=gws.parse_acl('allow x'))
    y = gws.Data(access=gws.parse_acl('allow y'))
    z = gws.Data(access=gws.parse_acl('allow z'))
    a = gws.Data(access=gws.parse_acl('allow a'))

    assert u.can_use(x) is False

    x.parent = y
    y.parent = a

    assert u.can_use(x) is True
