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
