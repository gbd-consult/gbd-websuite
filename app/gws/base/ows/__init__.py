import gws.types as t


#:export
class OwsOperation:
    def __init__(self):
        self.name = ''
        self.formats: t.List[str] = []
        self.get_url: t.Url = ''
        self.post_url: t.Url = ''
        self.parameters: dict = {}
