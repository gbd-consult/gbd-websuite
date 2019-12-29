### Request params and responses.

from .base import Optional
from ..data import Data


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
    mime: str
    content: str
    status: int


class FileResponse(Response):
    mime: str
    path: str
    status: int
    attachment_name: str
