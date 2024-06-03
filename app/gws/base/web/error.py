from typing import Optional

import gws
import werkzeug.exceptions

BadRequest = werkzeug.exceptions.BadRequest
Unauthorized = werkzeug.exceptions.Unauthorized
Forbidden = werkzeug.exceptions.Forbidden
NotFound = werkzeug.exceptions.NotFound
MethodNotAllowed = werkzeug.exceptions.MethodNotAllowed
NotAcceptable = werkzeug.exceptions.NotAcceptable
RequestTimeout = werkzeug.exceptions.RequestTimeout
Conflict = werkzeug.exceptions.Conflict
Gone = werkzeug.exceptions.Gone
LengthRequired = werkzeug.exceptions.LengthRequired
PreconditionFailed = werkzeug.exceptions.PreconditionFailed
RequestEntityTooLarge = werkzeug.exceptions.RequestEntityTooLarge
RequestURITooLarge = werkzeug.exceptions.RequestURITooLarge
UnsupportedMediaType = werkzeug.exceptions.UnsupportedMediaType
RequestedRangeNotSatisfiable = werkzeug.exceptions.RequestedRangeNotSatisfiable
ExpectationFailed = werkzeug.exceptions.ExpectationFailed
ImATeapot = werkzeug.exceptions.ImATeapot
UnprocessableEntity = werkzeug.exceptions.UnprocessableEntity
Locked = werkzeug.exceptions.Locked
PreconditionRequired = werkzeug.exceptions.PreconditionRequired
TooManyRequests = werkzeug.exceptions.TooManyRequests
RequestHeaderFieldsTooLarge = werkzeug.exceptions.RequestHeaderFieldsTooLarge
UnavailableForLegalReasons = werkzeug.exceptions.UnavailableForLegalReasons
InternalServerError = werkzeug.exceptions.InternalServerError
NotImplemented = werkzeug.exceptions.NotImplemented
BadGateway = werkzeug.exceptions.BadGateway
ServiceUnavailable = werkzeug.exceptions.ServiceUnavailable
GatewayTimeout = werkzeug.exceptions.GatewayTimeout
HTTPVersionNotSupported = werkzeug.exceptions.HTTPVersionNotSupported

HTTPException = werkzeug.exceptions.HTTPException


def from_exception(exc: Exception) -> HTTPException:
    """Convert generic errors to http errors."""

    if isinstance(exc, HTTPException):
        return exc

    e = None

    if isinstance(exc, gws.NotFoundError):
        e = NotFound()
    elif isinstance(exc, gws.ForbiddenError):
        e = Forbidden()
    elif isinstance(exc, gws.BadRequestError):
        e = BadRequest()
    elif isinstance(exc, gws.ResponseTooLargeError):
        e = Conflict()

    if e:
        gws.log.info(f'HTTPException: {e.code} cause={exc}')
    else:
        gws.log.exception()
        e = InternalServerError()

    e.__cause__ = exc
    return e
