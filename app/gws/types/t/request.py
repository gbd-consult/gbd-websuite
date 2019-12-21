### Request params and responses.

from .base import Optional
from .data import Data
from .auth import AuthUser
from .object import Object
from .map import ProjectObject

import werkzeug.wrappers


class Params(Data):
    projectUid: Optional[str]  #: project uid
    locale: Optional[str]  #: locale for this request


class NoParams(Data):
    pass


class ResponseError(Data):
    status: int
    info: str


class Response(Data):
    error: Optional[ResponseError]


class HttpResponse(Response):
    mimeType: str
    content: str
    status: int


class Request:
    environ: dict
    cookies: dict
    has_struct: bool
    expected_struct: str
    data: bytes
    params: dict
    kparams: dict
    post_data: str
    user: 'AuthUser'

    def response(self, content, mimetype: str, status=200) -> werkzeug.wrappers.Response:
        pass

    def struct_response(self, data, status=200) -> werkzeug.wrappers.Response:
        pass

    def env(self, key: str, default=None) -> str:
        pass

    def param(self, key: str, default=None) -> str:
        pass

    def kparam(self, key: str, default=None) -> str:
        pass

    def url_for(self, url: str) -> str:
        pass

    def require(self, klass: str, uid: str) -> Object:
        pass

    def require_project(self, uid: str) -> 'ProjectObject':
        pass

    def acquire(self, klass: str, uid: str) -> Object:
        pass

    def login(self, username: str, password: str):
        pass

    def logout(self):
        pass

    def auth_begin(self):
        pass

    def auth_commit(self, res: werkzeug.wrappers.Response):
        pass
