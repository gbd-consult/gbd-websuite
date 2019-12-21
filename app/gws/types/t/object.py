### Basic tree node object.

from .base import List, Tuple
from .data import Config, Props
from .auth import AuthUser
from .application import ApplicationObject


class Object:
    children: List['Object']
    config: Config
    klass: str
    parent: 'Object'
    root: 'Object'
    uid: str
    props: Props

    def is_a(self, klass):
        pass

    def initialize(self, cfg):
        pass

    def configure(self):
        pass

    def var(self, key, default=None, parent=False):
        pass

    def add_child(self, klass, cfg):
        pass

    def get_children(self, klass) -> List['Object']:
        pass

    def get_closest(self, klass) -> 'Object':
        pass

    def find_all(self, klass=None) -> List['Object']:
        pass

    def find_first(self, klass) -> 'Object':
        pass

    def find(self, klass, uid) -> List['Object']:
        pass

    def props_for(self, user: 'AuthUser') -> Props:
        pass


class RootObject(Object):
    application: 'ApplicationObject'
    all_types: dict
    all_objects: List['Object']
    shared_objects: dict

    def create(self, klass, cfg=None) -> 'Object':
        pass

    def validate_action(self, category: str, cmd: str, payload: dict) -> Tuple[str, str, dict]:
        pass
