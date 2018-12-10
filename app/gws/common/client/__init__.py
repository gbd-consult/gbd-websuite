import gws
import gws.types as t


class Config(t.Config):
    """client UI element"""

    access: t.Optional[t.Access]  #: access rights
    tag: str  #: element tag
    options: t.Optional[dict]  #: options for this element
    elements: t.Optional[t.List['Config']]  #: child elements of this element


class Props(t.Data):
    tag: str
    options: t.Optional[dict]
    elements: t.Optional[t.List['Props']]


class Object(gws.PublicObject):
    def configure(self):
        super().configure()
        for c in self.var('elements', []):
            self.add_child(Object, c)

    @property
    def props(self):
        els = self.get_children(Object)
        return {
            'tag': self.var('tag'),
            'options': self.var('options'),
            'elements': els or None,
        }
