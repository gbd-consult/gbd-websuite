from typing import Optional, cast

import gws
import gws.spec.runtime
import gws.test.util as u
from gws.base.auth.user import User


# @formatter:off

class H1(gws.Node): pass
class H2(gws.Node): pass

class A1(gws.Node): pass
class A2(gws.Node): pass

class X(gws.Node):
    def configure(self):
        u.log.write('X')

class Y(X):
    def configure(self):
        u.log.write('Y')

class Z(Y):
    def configure(self):
        u.log.write('Z')

# @formatter:on


@u.fixture(scope='function')
def root():
    r = u.gws_root('', defaults=False)

    r.specs.register_object(gws.ext.object.helper, 'H1', H1)
    r.specs.register_object(gws.ext.object.helper, 'H2', H2)

    r.specs.register_object(gws.ext.object.action, 'A1', A1)
    r.specs.register_object(gws.ext.object.action, 'A2', A2)

    r.specs.register_object(gws.ext.object.helper, 'X', X)
    r.specs.register_object(gws.ext.object.helper, 'Y', Y)
    r.specs.register_object(gws.ext.object.helper, 'Z', Z)

    yield r


def test_create(root: gws.Root):
    a1 = root.create(gws.ext.object.action, type='A1')
    a2 = root.create(gws.ext.object.action, type='A2')

    assert not root.configErrors
    assert root.find_all(gws.ext.object.action) == [a1, a2]


def test_create_shared(root: gws.Root):
    a1 = root.create_shared(gws.ext.object.action, type='A1', uid='x')
    a2 = root.create_shared(A1, uid='x')

    assert not root.configErrors
    assert a1 == a2


def test_create_child(root: gws.Root):
    a = root.create(A1)
    b1 = a.create_child(H1)
    b2 = a.create_child(H2)

    assert not root.configErrors
    assert a.children == [b1, b2]


def test_create_child_if_configured(root: gws.Root):
    a = root.create(A1)
    b1 = a.create_child_if_configured(H1, gws.Config(foo=123))
    b2 = a.create_child_if_configured(H1)

    assert not root.configErrors
    assert a.children == [b1]
    assert b2 is None


def test_auto_super_configure(root: gws.Root):
    u.log.reset()
    root.create(Z)
    assert u.log.get() == ['X', 'Y', 'Z']


def test_is_a(root: gws.Root):
    a = root.create(A1)

    assert a.is_a(gws.ext.object.action)
    assert a.is_a('gws.ext.object.action')
    assert a.is_a('gws.ext.object.action.A1')
    assert a.is_a(A1)

    assert a.is_a(gws.ext.object.helper) is False
    assert a.is_a('gws.ext.object.helper') is False


def test_find_all(root: gws.Root):
    a = root.create(A1)
    b = root.create(A1)

    a1 = a.create_child(H1)
    a2 = a.create_child(H2)
    a3 = a.create_child(A2)

    b1 = b.create_child(H1)
    b2 = b.create_child(H2)

    assert a.find_all(gws.ext.object.helper) == [a1, a2]
    assert a.find_all(gws.ext.object.action) == [a3]
    assert a.find_all(H1) == [a1]
    assert a.find_all(gws.ext.object.model) == []


def test_find_first(root: gws.Root):
    a = root.create(A1)

    a1 = a.create_child(H1)
    a2 = a.create_child(H2)
    a3 = a.create_child(A2)

    assert a.find_first(gws.ext.object.helper) == a1
    assert a.find_first(gws.ext.object.action) == a3
    assert a.find_first(gws.ext.object.model) is None

    assert root.find_first(gws.ext.object.action) == a


def test_find_closest(root: gws.Root):
    a = root.create(A1)

    a1 = a.create_child(H1)
    a2 = a1.create_child(H2)
    a3 = a2.create_child(A2)
    a4 = a3.create_child(H2)

    assert a4.find_closest(gws.ext.object.action) == a3
    assert a4.find_closest(gws.ext.object.helper) == a2
    assert a4.find_closest(H1) == a1


def test_find_ancestors(root: gws.Root):
    a = root.create(A1)

    a1 = a.create_child(H1)
    a2 = a1.create_child(H2)
    a3 = a2.create_child(A2)
    a4 = a3.create_child(H2)

    assert a4.find_ancestors(gws.ext.object.action) == [a3, a]
    assert a4.find_ancestors() == [a3, a2, a1, a]


def test_find_descendants(root: gws.Root):
    a = root.create(A1)

    a1 = a.create_child(H1)
    a2 = a1.create_child(H2)
    b1 = a.create_child(A2)
    b2 = b1.create_child(H2)
    b3 = b1.create_child(A2)

    assert a.find_descendants(gws.ext.object.action) == [b1, b3]
    assert a.find_descendants() == [a1, a2, b1, b2, b3]
