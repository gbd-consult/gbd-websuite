import gws
import gws.types as t


class ElementConfig(t.Config):
    """GWS client UI element configuration"""

    access: t.Optional[t.Access]  #: access rights
    tag: str  #: element tag


class Config(t.Config):
    """GWS client configuration"""

    access: t.Optional[t.Access]  #: access rights
    options: t.Optional[dict]  #: client options
    elements: t.Optional[t.List[ElementConfig]]  #: client UI elements


class ElementProps(t.Data):
    tag: str


class Props(t.Data):
    options: t.Optional[dict]
    elements: t.Optional[t.List[ElementProps]]


class Element(gws.PublicObject):
    @property
    def props(self):
        return {
            'tag': self.var('tag'),
        }


class Object(gws.PublicObject):
    def configure(self):
        super().configure()
        for c in self.var('elements', []):
            self.add_child(Element, c)

    @property
    def props(self):
        return {
            'options': self.var('options'),
            'elements': self.children,
        }
