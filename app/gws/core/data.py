
# basic data type

def is_data_object(x):
    return isinstance(x, Data)


class Data:
    """Basic data object"""

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __repr__(self):
        return repr(vars(self))

    def get(self, k, default=None):
        return vars(self).get(k, default)

    def set(self, k, v):
        return setattr(self, k, v)

    def update(self, *args, **kwargs):
        d = {}
        for a in args:
            if isinstance(a, dict):
                d.update(a)
            elif isinstance(a, Data):
                d.update(vars(a))
        d.update(kwargs)
        vars(self).update(d)


# getattr needs to be defined out of class, otherwise IDEA accepts all attributes

def _data_getattr(self, attr):
    if attr.startswith('_'):
        # do not use None fallback for special props
        raise AttributeError(attr)
    return None


setattr(Data, '__getattr__', _data_getattr)

