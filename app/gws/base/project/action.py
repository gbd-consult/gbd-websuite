"""Project information action."""

from typing import Optional

import gws
import gws.base.action
import gws.base.auth.user
import gws.lib.intl


@gws.ext.config.action('project')
class Config(gws.base.action.Config):
    """Returns the project configuration to the client."""

    pass


@gws.ext.props.action('project')
class Props(gws.base.action.Props):
    pass


class InfoResponse(gws.Response):
    """Response of the ``projectInfo`` command."""

    project: gws.ext.props.project
    """Project properties for the current user."""
    locale: gws.Locale
    """Locale for the request."""
    user: Optional[gws.base.auth.user.Props]
    """Current user, or ``None`` for a guest."""


@gws.ext.object.action('project')
class Object(gws.base.action.Object):
    """Project action, provides the project configuration to the client."""

    @gws.ext.command.api('projectInfo')
    def info(self, req: gws.WebRequester, p: gws.Request) -> InfoResponse:
        """Return the project properties, the locale and the current user."""

        project = req.user.require_project(p.projectUid)

        return InfoResponse(
            project=gws.props_of(project, req.user),
            locale=gws.lib.intl.locale(p.localeUid, project.localeUids),
            user=None if req.user.isGuest else gws.props_of(req.user, req.user),
        )
