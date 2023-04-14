import gws


class CommandNotFound(gws.Error):
    pass


class CommandForbidden(gws.Error):
    pass


class BadRequest(gws.Error):
    pass


class Props(gws.Props):
    type: str


class Config(gws.ConfigWithAccess):
    pass


class Object(gws.Node, gws.IAction):
    """Generic action object, the parent of all action objects."""

    def props(self, user):
        return gws.Data(type=self.extType)
