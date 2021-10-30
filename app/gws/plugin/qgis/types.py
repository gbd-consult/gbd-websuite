import gws
import gws.lib.metadata
import gws.lib.gis
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


class ProjectCaps(gws.Data):
    metadata: gws.lib.metadata.Metadata
    print_templates: t.List[PrintTemplate]
    properties: dict
    source_layers: t.List[gws.lib.gis.SourceLayer]
    supported_crs: t.List[gws.Crs]
    version: str
