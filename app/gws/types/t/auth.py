# type: ignore

### Authorization provider and user

from .base import List
from .data import Props
from .object import Object


class AuthProviderObject(Object):
    def authenticate_user(self, login: str, password: str, **kw) -> 'AuthUser':
        pass

    def get_user(self, user_uid: str) -> 'AuthUser':
        pass

    def unmarshal_user(self, user_uid: str, s: str) -> 'AuthUser':
        pass

    def marshal_user(self, user: 'AuthUser') -> str:
        pass


class AuthUser:
    display_name: str
    props: Props
    is_guest: bool
    full_uid: str

    def init_from_source(self, provider: AuthProviderObject, uid: str, roles: List[str] = None, attributes: dict = None):
        pass

    def init_from_cache(self, provider: AuthProviderObject, uid: str, roles: List[str], attributes: dict):
        pass

    def attribute(self, key, default=''):
        pass

    def can_use(self, obj: 'Object', parent: 'Object' = None) -> bool:
        return False
