import gws


class Config(gws.WithAccess):
    pass


class Props(gws.Props):
    enabled: bool = True


class Object(gws.Node):
    """Generic action object, the parent of all action objects."""

    @property
    def props(self):
        return Props(enabled=True, type=self.ext_type)
