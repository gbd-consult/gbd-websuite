"""Project information action."""

import gws.common.action
import gws.common.project
import gws.tools.intl
import gws.web.error

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Project information action"""
    pass


class InfoParams(t.Params):
    pass


class InfoResponse(t.Response):
    project: gws.common.project.Props
    locale: gws.tools.intl.Locale
    user: t.Optional[t.UserProps]


class Object(gws.common.action.Object):
    def api_info(self, req: t.IRequest, p: InfoParams) -> InfoResponse:
        """Return the project configuration"""

        project = req.require_project(p.projectUid)

        lo = gws.tools.intl.locale_data(p.locale)
        if not lo:
            lo = gws.tools.intl.locale_data(project.locales[0])

        return InfoResponse(
            project=project.props_for(req.user),
            locale=lo,
            user=None if req.user.is_guest else req.user.props)
