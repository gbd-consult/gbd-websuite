import gws


class Config(gws.WithAccess):
    pass


class Props(gws.Props):
    type: str


class Object(gws.Node):
    """Generic action object, the parent of all action objects."""

    def props_for(self, user):
        return Props(type=self.ext_type)
