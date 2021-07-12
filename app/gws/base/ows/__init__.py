import gws
import gws.types as t



class OwsOperation:
    def __init__(self):
        self.name = ''
        self.formats: t.List[str] = []
        self.get_url: gws.Url = ''
        self.post_url: gws.Url = ''
        self.parameters: dict = {}
