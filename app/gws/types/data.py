### Data objects

class Data:
    """Data object."""

    def __init__(self, d=None):
        if d:
            for k, v in d.items():
                setattr(self, str(k), v)

    def set(self, k, value):
        return setattr(self, k, value)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def as_dict(self):
        return vars(self)

    def extend(self, *args):
        d = {}
        for a in args:
            if not isinstance(a, dict):
                a = a.as_dict() if hasattr(a, 'as_dict') else None
            if a:
                d.update(a)
        vars(self).update(d)
        return self

    def __repr__(self):
        return repr(vars(self))


class Config(Data):
    """Configuration base type"""

    uid: str = ''  #: unique ID


class Props(Data):
    """Properties base type"""
    pass
