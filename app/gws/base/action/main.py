import gws


class Props(gws.Props):
    type: str


class Config(gws.ConfigWithAccess):
    pass


class Object(gws.Node, gws.IAction):
    """Generic action object, the parent of all action objects."""

    def props(self, user):
        return gws.Data(type=self.extType)

