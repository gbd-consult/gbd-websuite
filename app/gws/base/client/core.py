import gws
import gws.types as t


class ElementConfig(gws.ConfigWithAccess):
    """GWS client UI element configuration"""

    tag: str
    """element tag"""
    before: str = ''
    """insert before this tag"""
    after: str = ''
    """insert after this tag"""


class Config(gws.ConfigWithAccess):
    """GWS client configuration"""

    options: t.Optional[dict]
    """client options"""
    elements: t.Optional[list[ElementConfig]]
    """client UI elements"""
    addElements: t.Optional[list[ElementConfig]]
    """add elements to the parent element list"""
    removeElements: t.Optional[list[ElementConfig]]
    """remove elements from the parent element list"""


class ElementProps(gws.Data):
    tag: str


class Props(gws.Data):
    options: t.Optional[dict]
    elements: t.Optional[list[ElementProps]]


class Element(gws.Node):
    tag: str
    after: str
    before: str

    def configure(self):
        self.tag = self.cfg('tag')
        self.after = self.cfg('after')
        self.before = self.cfg('before')

    def props(self, user):
        return gws.Data(tag=self.tag)


class Object(gws.Client):
    options: dict
    elements: list[Element]

    def configure(self):
        app_client = gws.u.get(self.root.app, 'client')

        self.elements = self.create_children(Element, self._get_elements(app_client))

        self.options = gws.u.merge(
            app_client.options if app_client else {},
            self.cfg('options'))

    def props(self, user):
        return Props(
            options=self.options,
            elements=self.elements,
        )

    def _get_elements(self, app_client):
        elements = self.cfg('elements')
        if elements:
            return elements

        if not app_client:
            return []

        add = self.cfg('addElements', default=[])
        remove = self.cfg('removeElements', default=[])
        elements = list(app_client.elements)

        for c in add:
            n = self._find_element(elements, c.tag)
            if n >= 0:
                elements.pop(n)
            if c.before:
                n = self._find_element(elements, c.before)
                if n >= 0:
                    elements.insert(n, c)
            elif c.after:
                n = self._find_element(elements, c.after)
                if n >= 0:
                    elements.insert(n + 1, c)
            else:
                elements.append(c)

        remove_tags = [c.tag for c in remove]
        return [e for e in elements if e.tag not in remove_tags]

    def _find_element(self, elements, tag):
        for n, el in enumerate(elements):
            if el.tag == tag:
                return n
        return -1
