"""OWS-specific errors.

Each error class corresponds to an OWS error code, as defined in OGC standards.
"""

import gws
import gws.lib.xmlx
import gws.lib.image
import gws.lib.mime


class Error(gws.Error):
    """OWS error."""

    def __init__(self, *args):
        super().__init__(*args)
        self.code = self.__class__.__name__
        self.locator = self.args[0] if len(self.args) > 0 else ''
        self.message = self.args[1] if len(self.args) > 1 else ''
        # NB assume it's the user's fault by default
        self.status = _STATUS.get(self.code, 400)

    def to_xml_response(self, xmlns='ows') -> gws.ContentResponse:
        """Returns an XML response for this error.

        Args:
            xmlns: XML namespace to use (``ogc`` or ``ows``).
        """

        if xmlns == 'ows':
            # OWS ExceptionReport, as per OGC 06-121r9, 8.5
            xml = gws.lib.xmlx.tag(
                'ExceptionReport', {'xmlns': 'ows'},
                ('Exception', {'exceptionCode': self.code, 'locator': self.locator}, self.message)
            )

        elif xmlns == 'ogc':
            # OGC ServiceExceptionReport, as per OGC 06-042, H.2
            xml = gws.lib.xmlx.tag(
                'ServiceExceptionReport', {'xmlns': 'ogc'},
                ('ServiceException', {'exceptionCode': self.code, 'locator': self.locator}, self.message)
            )

        else:
            raise gws.Error(f'invalid {xmlns=}')

        return gws.ContentResponse(
            status=self.status,
            mime=gws.lib.mime.XML,
            content=xml.to_string(
                with_namespace_declarations=True,
                with_schema_locations=False,
            )
        )

    def to_image_response(self, mime='image/png') -> gws.ContentResponse:
        """Returns an image response for this error.

        Args:
            mime: Image mime type.
        """

        return gws.ContentResponse(
            status=self.status,
            mime=mime,
            content=gws.lib.image.error_pixel(mime),
        )


def from_exception(exc: Exception) -> Error:
    """Convert an Exception to the OWS Error."""

    if isinstance(exc, Error):
        return exc

    e = None

    if isinstance(exc, gws.NotFoundError):
        e = NotFound()
    elif isinstance(exc, gws.ForbiddenError):
        e = Forbidden()
    elif isinstance(exc, gws.BadRequestError):
        e = BadRequest()

    if e:
        gws.log.warning(f'OWS Exception: {e.code} cause={exc!r}')
    else:
        gws.log.exception()
        e = NoApplicableCode('', 'Internal Server Error')

    e.__cause__ = exc
    return e


# @formatter:off

# out extensions

class NotFound(Error): ...
class Forbidden(Error): ...
class BadRequest(Error): ...

# OGC 06-121r9
# Table 27 — Standard exception codes and meanings

class InvalidParameterValue(Error): ...
class InvalidUpdateSequence(Error): ...
class MissingParameterValue(Error): ...
class NoApplicableCode(Error): ...
class OperationNotSupported(Error): ...
class OptionNotSupported(Error): ...
class VersionNegotiationFailed(Error): ...

# OGC 06-042
# Table E.1 — Service exception codes

class CurrentUpdateSequence(Error): ...
class InvalidCRS(Error): ...
class InvalidDimensionValue(Error): ...
class InvalidFormat(Error): ...
class InvalidPoint(Error): ...
class LayerNotDefined(Error): ...
class LayerNotQueryable(Error): ...
class MissingDimensionValue(Error): ...
class StyleNotDefined(Error): ...

# OGC 07-057r7
# Table 20 — Exception codes for GetCapabilities operation
# Table 23 — Exception codes for GetTile operation

class PointIJOutOfRange(Error): ...
class TileOutOfRange(Error): ...

# OGC 09-025r1
# Table 3 — WFS exception codes

class CannotLockAllFeatures(Error): ...
class DuplicateStoredQueryIdValue(Error): ...
class DuplicateStoredQueryParameterName(Error): ...
class FeaturesNotLocked(Error): ...
class InvalidLockId(Error): ...
class InvalidValue(Error): ...
class LockHasExpired(Error): ...
class OperationParsingFailed(Error): ...
class OperationProcessingFailed(Error): ...
class ResponseCacheExpired(Error): ...

# OGC 06-121r9  8.6 HTTP STATUS codes for OGC Exceptions

_STATUS = dict(
    OperationNotSupported=501,
    MissingParameterValue=400,
    InvalidParameterValue=400,
    VersionNegotiationFailed=400,
    InvalidUpdateSequence=400,
    OptionNotSupported=501,
    NoApplicableCode=500,

    NotFound=404,
    Forbidden=403,
    BadRequest=400,

)
