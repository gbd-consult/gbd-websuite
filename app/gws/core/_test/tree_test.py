from gws.core.types import Props, IUser
from gws.core.tree import BaseObject, Object, RootObject


def test_is_a():
    class Foo(BaseObject):
        pass

    assert Foo().class_name.endswith('.Foo')
    assert Foo('xxx').is_a('xxx')


def test_props_and_access():
    class Open(BaseObject):
        @property
        def props(self):
            return Props(name='open')

    class Secret(BaseObject):
        @property
        def props(self):
            return Props(name='secret')

    class NormalUser(IUser):
        def can_use(self, obj):
            return 'Open' in obj.class_name or 'Main' in obj.class_name

    class AdminUser(IUser):
        def can_use(self, obj):
            return True

    class Main(BaseObject):
        @property
        def props(self):
            return Props(
                open=Open(),
                secret=Secret(),
                var=42
            )

    p = Main().props_for(NormalUser())
    assert vars(p) == {'open': {'name': 'open'}, 'var': 42}

    p = Main().props_for(AdminUser())
    assert vars(p) == {'open': {'name': 'open'}, 'secret': {'name': 'secret'}, 'var': 42}


def test_configure_auto_inheritance():
    log = []

    class A(Object):
        def configure(self):
            log.append('A')

    class B(A):
        def configure(self):
            log.append('B')

    class C(B):
        def configure(self):
            log.append('C')

    c = RootObject().create_object(C, {})
    assert log == ['A', 'B', 'C']


def test_create_child():
    class B(Object):
        def configure(self):
            self.foo = 'B'

    class C(Object):
        def configure(self):
            self.foo = 'C'

    class A(Object):
        def configure(self):
            self.create_child(B, {})
            self.create_child(C, {})

    a = RootObject().create_object(A, {})
    assert [c.foo for c in a.children] == ['B', 'C']
