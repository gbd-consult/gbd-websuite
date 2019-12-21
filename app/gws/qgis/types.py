import gws.types as t


class PrintTemplateElement:
    def __init__(self):
        self.tag = ''
        self.attrs = {}
        self.type = ''
        # more auto props....


class PrintTemplate:
    def __init__(self):
        self.title = ''
        self.index = 0
        self.attrs = {}
        self.elements: t.List[PrintTemplateElement] = []


class SourceLayer(t.SourceLayer):
    pass


class ProviderObject(t.OwsProviderObject):
    extent: t.Extent
    legend_params: dict
    path: str
    print_templates: t.List[PrintTemplate]
    properties: dict
