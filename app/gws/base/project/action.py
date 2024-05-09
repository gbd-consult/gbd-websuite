from typing import Optional

import gws
import gws.base.action
import gws.base.auth.user
import gws.lib.intl

from . import core

gws.ext.new.action('project')


class Config(gws.base.action.Config):
    pass


class Props(gws.base.action.Props):
    pass


class InfoResponse(gws.Response):
    project: gws.ext.props.project
    locale: gws.Locale
    user: Optional[gws.base.auth.user.Props]


class Object(gws.base.action.Object):
    """Project information action"""

    @gws.ext.command.api('projectInfo')
    def info(self, req: gws.WebRequester, p: gws.Request) -> InfoResponse:
        """Return the project configuration"""

        project = req.user.require_project(p.projectUid)

        locale_uid = p.localeUid
        if locale_uid not in project.localeUids:
            locale_uid = project.localeUids[0]

        return InfoResponse(
            project=gws.props_of(project, req.user),
            locale=gws.lib.intl.locale(locale_uid),
            user=None if req.user.isGuest else gws.props_of(req.user, req.user))
