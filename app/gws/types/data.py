class Data:
    def __init__(self, d=None):
        if d:
            for k, v in d.items():
                setattr(self, str(k), v)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def as_dict(self):
        return vars(self)
