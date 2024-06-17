from typing import Optional, cast

import gws
import gws.spec.runtime
import gws.test.util as u
from gws.base.auth.user import User


@u.fixture(scope='function')
def root():
    yield u.gws_root('')


##

def _register(root, *classes):
    for cls in classes:
        root.specs.register_object(gws.ext.object.helper, cls.__name__, cls)


def test_create(root: gws.Root):
    class A(gws.Node):
        pass

    class B(gws.Node):
        pass

    _register(root, A, B)

    a = root.create(gws.ext.object.helper, type='A', uid='a')
    b = root.create(gws.ext.object.helper, type='B', uid='b')

    assert root.configErrors == []
    assert root.find_all(gws.ext.object.helper) == [a, b]


def test_create_shared(root: gws.Root):
    class A(gws.Node):
        pass

    _register(root, A)

    a1 = root.create_shared(gws.ext.object.helper, type='A', uid='a')
    a2 = root.create_shared(gws.ext.object.helper, type='A', uid='a')

    assert root.configErrors == []
    assert a1 == a2


def test_create_child(root: gws.Root):
    class A(gws.Node):
        pass

    class B(gws.Node):
        pass

    _register(root, A, B)

    a = root.create(A)
    b1 = a.create_child(B)
    b2 = a.create_child(B)

    assert root.configErrors == []
    assert a.children == [b1, b2]


def test_auto_super_configure(root: gws.Root):
    log = []

    class A(gws.Node):
        def configure(self):
            log.append('A')

    class B(A):
        def configure(self):
            log.append('B')

    class C(B):
        def configure(self):
            log.append('C')

    _register(root, A, B, C)

    root.create(C)

    assert log == ['A', 'B', 'C']
