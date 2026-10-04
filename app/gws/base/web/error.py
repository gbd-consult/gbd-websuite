"""HTTP exceptions and conversion of GWS errors to HTTP errors."""

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
    """Convert an exception to an HTTP exception.

    HTTP exceptions are returned as is. ``gws.NotFoundError``, ``gws.ForbiddenError``,
    ``gws.BadRequestError``, ``gws.TooManyRequestsError`` and ``gws.ResponseTooLargeError``
    are converted to 404, 403, 400, 429 and 409 and logged as warnings. Other exceptions
    are logged with the traceback and converted to 500. The original exception is
    set as the cause of the new one.

    Args:
        exc: An exception.

    Returns:
        An HTTP exception.
    """

    if isinstance(exc, HTTPException):
        return exc

    e = None

    if isinstance(exc, gws.NotFoundError):
        e = NotFound()
    elif isinstance(exc, gws.ForbiddenError):
        e = Forbidden()
    elif isinstance(exc, gws.BadRequestError):
        e = BadRequest()
    elif isinstance(exc, gws.TooManyRequestsError):
        e = TooManyRequests(retry_after=exc.retryAfter or None)
    elif isinstance(exc, gws.ResponseTooLargeError):
        e = Conflict()

    if e:
        gws.log.warning(f'HTTPException: {e.code} cause={exc!r}')
    else:
        gws.log.exception()
        e = InternalServerError()

    e.__cause__ = exc
    return e
