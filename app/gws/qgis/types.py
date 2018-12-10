import gws.types as t


class PrintTemplateElement:
    def __init__(self):
        self.tag = ''
        self.attrs = {}
        # more auto props....


class PrintTemplate:
    def __init__(self):
        self.title = ''
        self.index = 0
        self.attrs = {}
        self.elements: t.List[PrintTemplateElement] = []


class WmsService(t.Service):
    def __init__(self):
        self.type = 'QGIS/WMS'
        self.path = ''
        self.version = '1.3.0'  # as of QGIS 3.4


class SourceLayer(t.SourceLayer):
    pass
