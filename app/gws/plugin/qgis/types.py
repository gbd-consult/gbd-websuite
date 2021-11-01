import gws
import gws.lib.gis
import gws.lib.metadata
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
