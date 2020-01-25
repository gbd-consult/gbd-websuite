import gws.common.project
import gws.web
import gws.tools.intl
import gws.types as t


class Config(t.WithTypeAndAccess):
    """Project information action"""
    pass


class InfoParams(t.Params):
    pass


class InfoResponse(t.Response):
    project: gws.common.project.Props
    localeData: gws.tools.intl.LocaleData
    user: t.Optional[t.UserProps]


class Object(gws.ActionObject):
    def api_info(self, req: t.IRequest, p: InfoParams) -> InfoResponse:
        """Return the project configuration"""

        project = req.require_project(p.projectUid)

        ld = gws.tools.intl.locale_data(p.locale)
        if not ld:
            ld = gws.tools.intl.locale_data(project.locales[0])

        return InfoResponse(
            project=project.props_for(req.user),
            localeData=ld,
            user=None if req.user.is_guest else req.user.props)
