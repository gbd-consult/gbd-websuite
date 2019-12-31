### Data objects

class Data:
    """Data object."""

    def __init__(self, *args, **kwargs):
        self._extend(args, kwargs)

    def set(self, k, value):
        return setattr(self, k, value)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def as_dict(self):
        return vars(self)

    def extend(self, *args, **kwargs):
        self._extend(args, kwargs)
        return self

    def __repr__(self):
        return repr(vars(self))

    def _extend(self, args, kwargs):
        d = {}
        for a in args:
            d.update(a)
        d.update(kwargs)
        vars(self).update(d)


class Config(Data):
    """Configuration base type"""

    uid: str = ''  #: unique ID


class Props(Data):
    """Properties base type"""
    pass
