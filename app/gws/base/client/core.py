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

    options: t.Optional[t.Dict] 
    """client options"""
    elements: t.Optional[t.List[ElementConfig]] 
    """client UI elements"""
    addElements: t.Optional[t.List[ElementConfig]] 
    """add elements to the parent element list"""
    removeElements: t.Optional[t.List[ElementConfig]] 
    """remove elements from the parent element list"""


class ElementProps(gws.Data):
    tag: str


class Props(gws.Data):
    options: t.Optional[dict]
    elements: t.Optional[t.List[ElementProps]]


class Element(gws.Node):
    def props(self, user):
        return gws.Data(tag=self.var('tag'))


class Object(gws.Node, gws.IClient):
    options: dict
    elements: t.List[Element]

    def configure(self):
        app_client = gws.get(self.root.app, 'client')

        self.elements = self.create_children(Element, self._get_elements(app_client))

        self.options = gws.merge(
            app_client.options if app_client else {},
            self.var('options'))

    def props(self, user):
        return gws.Data(
            options=self.options,
            elements=[e.props(user) for e in self.elements]
        )

    def _get_elements(self, app_client):
        elements = self.var('elements')
        if elements:
            return elements

        if not app_client:
            return []

        add = self.var('addElements', default=[])
        remove = self.var('removeElements', default=[])
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