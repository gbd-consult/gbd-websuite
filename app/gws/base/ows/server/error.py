"""OWS exceptions."""

import gws
import gws.lib.xmlx as xmlx
import gws.lib.image
import gws.lib.mime


class Error(gws.Error):
    """OWS error.

    The class name is used as the OWS exception code. The HTTP status is
    derived from the code, 400 by default.
    """

    def __init__(self, *args):
        """Create an error.

        Args:
            *args: The locator (usually the name of the offending parameter) and the error message, both optional.
        """
        super().__init__(*args)
        self.code = self.__class__.__name__
        self.locator = self.args[0] if len(self.args) > 0 else ''
        self.message = self.args[1] if len(self.args) > 1 else ''
        # NB assume it's the user's fault by default
        self.status = _STATUS.get(self.code, 400)

    def to_xml_response(self, xmlns='ows') -> gws.ContentResponse:
        """Create an XML response for this error.

        Args:
            xmlns: Format to use: ``ows`` for an OWS ``ExceptionReport``,
                ``ogc`` for an OGC ``ServiceExceptionReport``.

        Returns:
            The XML response.

        Raises:
            ``gws.Error``: If ``xmlns`` is invalid.
        """

        if xmlns == 'ows':
            # OWS ExceptionReport, as per OGC 06-121r9, 8.5
            ns = xmlx.namespace.c.OWS_11
            xml = xmlx.tag(
                'ExceptionReport',
                xmlx.tag(
                    'Exception',
                    {'exceptionCode': self.code, 'locator': self.locator},
                    self.message,
                ),
            )

        elif xmlns == 'ogc':
            # OGC ServiceExceptionReport, as per OGC 06-042, H.2
            ns = xmlx.namespace.c.OGC
            xml = xmlx.tag(
                'ServiceExceptionReport',
                xmlx.tag(
                    'ServiceException',
                    {'code': self.code},
                    self.message,
                ),
            )

        else:
            raise gws.Error(f'invalid {xmlns=}')

        return gws.ContentResponse(
            status=self.status,
            mimeType=gws.lib.mime.XML,
            content=xml.to_string(
                gws.XmlOptions(
                    defaultNamespace=ns,
                    withXmlDeclaration=True,
                    withNamespaceDeclarations=True,
                )
            ),
        )

    def to_image_response(self, mime_type='image/png') -> gws.ContentResponse:
        """Create an image response for this error.

        The image is a single error pixel.

        Args:
            mime_type: Image mime type.

        Returns:
            The image response.
        """

        return gws.ContentResponse(
            status=self.status,
            mimeType=mime_type,
            content=gws.lib.image.error_pixel(mime_type),
        )


def from_exception(exc: Exception) -> Error:
    """Convert an exception to an OWS error.

    ``gws.NotFoundError``, ``gws.ForbiddenError`` and ``gws.BadRequestError`` are
    mapped to the corresponding errors, any other exception is logged and becomes
    ``NoApplicableCode``. OWS errors are returned unchanged.

    Args:
        exc: An exception.

    Returns:
        The OWS error. A converted error has ``exc`` as its cause.
    """

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


class NotFound(Error):
    """The requested resource was not found (HTTP 404)."""


class Forbidden(Error):
    """Access to the requested resource is forbidden (HTTP 403)."""


class BadRequest(Error):
    """The request is invalid (HTTP 400)."""


# OGC 06-121r9
# Table 27 — Standard exception codes and meanings


class InvalidParameterValue(Error):
    """A parameter has an invalid value."""


class InvalidUpdateSequence(Error):
    """The update sequence is greater than the current one."""


class MissingParameterValue(Error):
    """A required parameter is missing."""


class NoApplicableCode(Error):
    """No other exception code applies, e.g. an internal server error."""


class OperationNotSupported(Error):
    """The requested operation is not supported."""


class OptionNotSupported(Error):
    """The requested option is not supported."""


class VersionNegotiationFailed(Error):
    """None of the requested versions is supported."""


# OGC 06-042
# Table E.1 — Service exception codes


class CurrentUpdateSequence(Error):
    """The update sequence equals the current one."""


class InvalidCRS(Error):
    """The requested CRS is not supported."""


class InvalidDimensionValue(Error):
    """A dimension parameter has an invalid value."""


class InvalidFormat(Error):
    """The requested format is not supported."""


class InvalidPoint(Error):
    """The point coordinates are invalid."""


class LayerNotDefined(Error):
    """The requested layer does not exist."""


class LayerNotQueryable(Error):
    """The requested layer cannot be queried."""


class MissingDimensionValue(Error):
    """A required dimension value is missing."""


class StyleNotDefined(Error):
    """The requested style does not exist."""


# OGC 07-057r7
# Table 20 — Exception codes for GetCapabilities operation
# Table 23 — Exception codes for GetTile operation


class PointIJOutOfRange(Error):
    """The pixel coordinates are outside the tile."""


class TileOutOfRange(Error):
    """The tile row or column is out of range."""


# OGC 09-025r1
# Table 3 — WFS exception codes


class CannotLockAllFeatures(Error):
    """Not all requested features could be locked."""


class DuplicateStoredQueryIdValue(Error):
    """A stored query with this id already exists."""


class DuplicateStoredQueryParameterName(Error):
    """A stored query parameter name is used more than once."""


class FeaturesNotLocked(Error):
    """The features to modify are not locked."""


class InvalidLockId(Error):
    """The lock id is invalid."""


class InvalidValue(Error):
    """A value is invalid."""


class LockHasExpired(Error):
    """The lock has expired."""


class OperationParsingFailed(Error):
    """The request could not be parsed."""


class OperationProcessingFailed(Error):
    """The request could not be processed."""


class ResponseCacheExpired(Error):
    """The cached response has expired."""


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
