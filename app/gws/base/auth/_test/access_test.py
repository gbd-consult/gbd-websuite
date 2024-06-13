from typing import Optional, cast

import gws
import gws.test.util as u

from gws.base.auth import method, provider, user

mock_provider = cast(gws.AuthProvider, gws.Data(uid='mock_provider'))
mock_user = user.from_args(mock_provider, localUid='mock', roles=['a', 'b'])

root = u.gws_root()


def make_obj(acl):
    return root.create(gws.Node, permissions=gws.Data(read=gws.u.parse_acl(acl)))


def test_access_direct():
    obj = make_obj('allow a')
    assert mock_user.can_use(obj) is True

    obj = make_obj('deny a')
    assert mock_user.can_use(obj) is False

    obj = make_obj('deny x, allow b')
    assert mock_user.can_use(obj) is True

    obj = make_obj('deny x, deny b')
    assert mock_user.can_use(obj) is False

    obj = make_obj('deny x')
    assert mock_user.can_use(obj) is False


def test_access_context():
    x = make_obj('allow x')
    y = make_obj('allow y')
    z = make_obj('allow z')
    a = make_obj('allow a')

    assert mock_user.can_use(x) is False
    assert mock_user.can_use(x, y, z) is False
    assert mock_user.can_use(x, y, z, a) is True


def test_access_context_and_parent():
    x = make_obj('allow x')
    y = make_obj('allow y')
    z = make_obj('allow z')
    a = make_obj('allow a')

    assert mock_user.can_use(x) is False

    x.parent = y
    y.parent = a

    assert mock_user.can_use(x) is True
