import gws
import gws.spec.runtime
import gws.lib.test as test
from gws.base.auth.user import User
import gws.types as t


class MockSpecs(gws.ISpecRuntime):
    def __init__(self, *classes):
        self.classes = classes

    def parse_classref(self, classref):
        return gws.spec.runtime.Object.parse_classref(t.cast(gws.spec.runtime.Object, self), classref)

    def get_class(self, classref, ext_type=None):
        cls, name, ext_name = self.parse_classref(classref)

        if cls:
            return cls

        if ext_name:
            ext_name += '.' + (ext_type or gws.spec.core.DEFAULT_VARIANT_TAG)

        for c in self.classes:
            if name and c.__name__ == name:
                return c
            if ext_name and c.extName == ext_name:
                return c


def _root(*classes):
    return gws.create_root_object(MockSpecs(*classes))


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

def test_create():
    gws.ext.new.helper('A')

    class A(gws.Node):
        pass

    gws.ext.new.helper('B')

    class B(gws.Node):
        pass

    root = _root(A, B)
    a = root.create(gws.ext.object.helper, config={'type': 'A', 'uid': 'a'})
    b = root.create(gws.ext.object.helper, config={'type': 'B', 'uid': 'b'})

    assert root.configErrors == []
    assert root.find_all(gws.ext.object.helper) == [a, b]


def test_create_shared():
    gws.ext.new.helper('A')

    class A(gws.Node):
        pass

    root = _root(A)
    a1 = root.create_shared(gws.ext.object.helper, config={'type': 'A', 'uid': 'a'})
    a2 = root.create_shared(gws.ext.object.helper, config={'type': 'A', 'uid': 'a'})

    assert root.configErrors == []
    assert a1 == a2


def test_create_child():
    class A(gws.Node):
        pass

    class B(gws.Node):
        pass

    root = _root(A, B)

    a = root.create(A)
    b1 = a.create_child(B)
    b2 = a.create_child(B)

    assert root.configErrors == []
    assert a.children == [b1, b2]


def test_auto_super_configure():
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

    root = _root(C)
    root.create('C')

    assert log == ['A', 'B', 'C']

#
#
# def test_is_a():
#     class A(gws.Object):
#         pass
#
#     class B(A):
#         pass
#
#     a, b = _objects(A, B)
#
#     assert a.is_a(A) is True
#     assert b.is_a(A) is True
#
#     class B1(gws.Node):
#         pass
#
#     class B2(B1):
#         pass
#
#     b1, b2 = _nodes(B1, B2)
#
#     assert b1.is_a(B1) is True
#     assert b2.is_a(B1) is True
#
#     assert b1.is_a('YES') is True
#
#
# def test_basic_access():
#     class A(gws.Object):
#         pass
#
#     a, = _objects(A)
#
#     a.access = _access('X allow Y deny')
#
#     assert _user('X').can_use(a) is True
#     assert _user('Y').can_use(a) is False
#     assert _user('Z').can_use(a) is False
#
#
# def test_access_with_parent():
#     class A(gws.Object):
#         pass
#
#     class B(gws.Object):
#         pass
#
#     a, b = _objects(A, B)
#
#     a.access = _access('X allow Y deny')
#     b.access = _access('Z allow')
#
#     b.parent = a
#
#     assert _user('X').can_use(b) is True
#     assert _user('Y').can_use(b) is False
#     assert _user('Z').can_use(b) is True
#
#
# def test_access_with_explicit_parent():
#     class A(gws.Object):
#         pass
#
#     class B(gws.Object):
#         pass
#
#     a, b = _objects(A, B)
#
#     a.access = _access('X allow')
#
#     assert _user('X').can_use(b) is False
#     assert _user('X').can_use(b, context=a) is True
#
#
# def test_basic_props():
#     class A(gws.Object):
#         def props(self, user):
#             return {'b': b, 'c': c}
#
#     class B(gws.Object):
#         def props(self, user):
#             return {'me': 'B'}
#
#     class C(gws.Object):
#         def props(self, user):
#             return {'me': 'C'}
#
#     a, b, c = _objects(A, B, C)
#
#     a.access = _access('X allow')
#
#     b.parent = a
#     c.parent = a
#
#     assert test.dict_of(gws.props(a, _user('X'))) == {'b': {'me': 'B'}, 'c': {'me': 'C'}}
#
#
# def test_props_with_access():
#     class A(gws.Object):
#         def props(self, user):
#             return {'b': b, 'c': c}
#
#     class B(gws.Object):
#         def props(self, user):
#             return {'me': 'B'}
#
#     class C(gws.Object):
#         def props(self, user):
#             return {'me': 'C'}
#
#     a, b, c = _objects(A, B, C)
#
#     a.access = _access('X allow')
#     b.access = _access('X deny')
#
#     b.parent = a
#     c.parent = a
#
#     assert test.dict_of(gws.props(a, _user('X'))) == {'c': {'me': 'C'}}
#
#
# def test_props_with_implicit_access():
#     class A(gws.Object):
#         def props(self, user):
#             return {'b': b, 'c': c}
#
#     class B(gws.Object):
#         def props(self, user):
#             return {'me': 'B'}
#
#     class C(gws.Object):
#         def props(self, user):
#             return {'d': d}
#
#     class D(gws.Object):
#         def props(self, user):
#             return {'me': 'D'}
#
#     a, b, c, d = _objects(A, B, C, D)
#
#     a.access = _access('X allow')
#     b.access = _access('X deny')
#
#     assert test.dict_of(gws.props(a, _user('X'))) == {'c': {'d': {'me': 'D'}}}
#
#
# #
# #
# # def test_create_child():
# #     class B(Object):
# #         def configure(self):
# #             self.foo = 'B'
# #
# #     class C(Object):
# #         def configure(self):
# #             self.foo = 'C'
# #
# #     class A(Object):
# #         def configure(self):
# #             self.create_child(B, {})
# #             self.create_child(C, {})
# #
# #     a = gws.Root().create_object(A, {})
# #     assert [c.foo for c in a.children] == ['B', 'C']
