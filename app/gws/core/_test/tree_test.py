import gws
import gws.spec.runtime
import gws.lib.test as test
from gws.base.auth.user import User
from types import MethodType


@test.fixture(scope='module', autouse=True)
def configuration():
    test.setup()
    yield
    test.teardown()


class MockSpecs:
    def __init__(self, classes):
        self.classes = {
            cls.__name__.rpartition('.')[-1]: cls
            for cls in classes
        }

    def real_class_names(self, class_name):
        return [class_name]

    def object_descriptor(self, class_name):
        c = class_name.rpartition('.')[-1]
        return gws.ExtObjectDescriptor(
            class_ptr=self.classes[c],
            ext_category='',
            ext_type='',
            ident='',
            module_name='',
            module_path='',
            name='',
        )

    def is_a(self, class_name, pattern):
        return pattern == 'YES'


def _objects(*classes):
    return [cls() for cls in classes]


def _nodes(*classes):
    root = gws.create_root_object(MockSpecs(classes))
    return [root.create(cls) for cls in classes]


def _user(role):
    provider = gws.Data(uid='_')
    return User(provider, role, [role], {})


def _access(words):
    w = words.split()
    return [
        gws.Access(role=w[i], type=w[i + 1])
        for i in range(0, len(w), 2)
    ]


##

def test_is_a():
    class A(gws.Object):
        pass

    class B(A):
        pass

    a, b = _objects(A, B)

    assert a.is_a(A) is True
    assert b.is_a(A) is True

    class B1(gws.Node):
        pass

    class B2(B1):
        pass

    b1, b2 = _nodes(B1, B2)

    assert b1.is_a(B1) is True
    assert b2.is_a(B1) is True

    assert b1.is_a('YES') is True


def test_basic_access():
    class A(gws.Object):
        pass

    a, = _objects(A)

    a.access = _access('X allow Y deny')

    assert _user('X').can_use(a) is True
    assert _user('Y').can_use(a) is False
    assert _user('Z').can_use(a) is False


def test_access_with_parent():
    class A(gws.Object):
        pass

    class B(gws.Object):
        pass

    a, b = _objects(A, B)

    a.access = _access('X allow Y deny')
    b.access = _access('Z allow')

    b.parent = a

    assert _user('X').can_use(b) is True
    assert _user('Y').can_use(b) is False
    assert _user('Z').can_use(b) is True


def test_access_with_explicit_parent():
    class A(gws.Object):
        pass

    class B(gws.Object):
        pass

    a, b = _objects(A, B)

    a.access = _access('X allow')

    assert _user('X').can_use(b) is False
    assert _user('X').can_use(b, context=a) is True


def test_basic_props():
    class A(gws.Object):
        def props(self, user):
            return {'b': b, 'c': c}

    class B(gws.Object):
        def props(self, user):
            return {'me': 'B'}

    class C(gws.Object):
        def props(self, user):
            return {'me': 'C'}

    a, b, c = _objects(A, B, C)

    a.access = _access('X allow')

    b.parent = a
    c.parent = a

    assert test.dict_of(gws.props(a, _user('X'))) == {'b': {'me': 'B'}, 'c': {'me': 'C'}}


def test_props_with_access():
    class A(gws.Object):
        def props(self, user):
            return {'b': b, 'c': c}

    class B(gws.Object):
        def props(self, user):
            return {'me': 'B'}

    class C(gws.Object):
        def props(self, user):
            return {'me': 'C'}

    a, b, c = _objects(A, B, C)

    a.access = _access('X allow')
    b.access = _access('X deny')

    b.parent = a
    c.parent = a

    assert test.dict_of(gws.props(a, _user('X'))) == {'c': {'me': 'C'}}


def test_props_with_implicit_access():
    class A(gws.Object):
        def props(self, user):
            return {'b': b, 'c': c}

    class B(gws.Object):
        def props(self, user):
            return {'me': 'B'}

    class C(gws.Object):
        def props(self, user):
            return {'d': d}

    class D(gws.Object):
        def props(self, user):
            return {'me': 'D'}

    a, b, c, d = _objects(A, B, C, D)

    a.access = _access('X allow')
    b.access = _access('X deny')

    assert test.dict_of(gws.props(a, _user('X'))) == {'c': {'d': {'me': 'D'}}}


def test_configure_auto_inheritance():
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

    _nodes(C)

    assert log == ['A', 'B', 'C']
#
#
# def test_create_child():
#     class B(Object):
#         def configure(self):
#             self.foo = 'B'
#
#     class C(Object):
#         def configure(self):
#             self.foo = 'C'
#
#     class A(Object):
#         def configure(self):
#             self.create_child(B, {})
#             self.create_child(C, {})
#
#     a = gws.Root().create_object(A, {})
#     assert [c.foo for c in a.children] == ['B', 'C']
