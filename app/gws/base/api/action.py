import gws


class Props(gws.Props):
    type: str


class Config(gws.WithAccess):
    pass


class Object(gws.Node, gws.IAction):
    """Generic action object, the parent of all action objects."""

    def props_for(self, user):
        return gws.Data(type=self.ext_type)
