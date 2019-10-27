class Data:
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

    def __repr__(self):
        return repr(vars(self))
