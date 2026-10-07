"""QField Cloud API action."""

from typing import Optional

import gws
import gws.base.auth
import gws.base.action
import gws.lib.mime
import gws.lib.jsonx

from . import core, action_base, action_handler


@gws.ext.config.action('qfieldcloud')
class Config(gws.ConfigWithAccess):
    """Endpoint for QField clients that emulates the QFieldCloud API."""

    projects: list[core.ProjectConfig]
    """Projects offered to QField clients."""
    auth: Optional[gws.base.auth.method.Config]
    """Token authentication method for QField clients."""


@gws.ext.props.action('qfieldcloud')
class Props(gws.base.action.Props):
    pass


@gws.ext.object.action('qfieldcloud')
class Object(action_base.BaseAction):
    """QField Cloud API action.

    Emulates the QFieldCloud API for the QField app: authenticates clients by
    token, lists the configured QField projects, creates packages in background
    jobs and applies deltas and file uploads from the devices.
    """

    @gws.ext.command.raw('qfieldcloudApi')
    def raw_request(self, req: gws.WebRequester, p: gws.Request) -> gws.ContentResponse:
        """Handle a QFieldCloud API request."""
        try:
            return self.get_handler().handle(req, p)
        except gws.NotFoundError as exc:
            return _error_response(404, 'object_not_found', exc)
        except gws.AuthenticationError as exc:
            return _error_response(401, 'authentication_failed', exc)
        except gws.ForbiddenError as exc:
            return _error_response(403, 'permission_denied', exc)
        except gws.BadRequestError as exc:
            return _error_response(400, 'validation_error', exc)

    def get_handler(self) -> action_handler.Handler:
        """Return a new request handler. Override to use a custom handler.

        Returns:
            Request handler.
        """
        return action_handler.Handler(self)


##


def _error_response(status: int, code: str, exc: Exception) -> gws.ContentResponse:
    """Log an error and return a JSON error response."""
    gws.log.warning(f'qfieldcloudApi: {status} {code} cause={exc!r}')
    return gws.ContentResponse(
        status=status,
        content=gws.lib.jsonx.to_string({'code': code}),
        mimeType=gws.lib.mime.JSON,
    )
