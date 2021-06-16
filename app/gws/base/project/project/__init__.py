"""Project information action."""

import gws.base.action
import gws.base.project
import gws.lib.intl
import gws.web.error

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Project information action"""
    pass


class InfoParams(t.Params):
    pass


class InfoResponse(t.Response):
    project: gws.base.project.Props
    locale: gws.lib.intl.Locale
    user: t.Optional[t.UserProps]


class Object(gws.base.action.Object):
    def api_info(self, req: t.IRequest, p: InfoParams) -> InfoResponse:
        """Return the project configuration"""

        project = req.require_project(p.projectUid)

        locale_uid = p.localeUid
        if locale_uid not in project.locale_uids:
            locale_uid = project.locale_uids[0]

        return InfoResponse(
            project=project.props_for(req.user),
            locale=gws.lib.intl.locale(locale_uid),
            user=None if req.user.is_guest else req.user.props)
