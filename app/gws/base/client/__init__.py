import gws
import gws.types as t


class ElementConfig(gws.WithAccess):
    """GWS client UI element configuration"""

    tag: str  #: element tag
    before: str = ''  #: insert before this tag
    after: str = ''  #: insert after this tag


class Config(gws.WithAccess):
    """GWS client configuration"""

    options: t.Optional[dict]  #: client options
    elements: t.Optional[t.List[ElementConfig]]  #: client UI elements
    addElements: t.Optional[t.List[ElementConfig]]  #: add elements to the parent element list
    removeElements: t.Optional[t.List[ElementConfig]]  #: remove elements from the parent element list


class ElementProps(gws.Data):
    tag: str


class Props(gws.Data):
    options: t.Optional[dict]
    elements: t.Optional[t.List[ElementProps]]


class Element(gws.Node):
    def props_for(self, user):
        return ElementProps(tag=self.var('tag'))


class Object(gws.Node, gws.IClient):
    options: dict
    elements: t.List[Element]

    def props_for(self, user):
        return Props(options=self.options, elements=self.elements)

    def configure(self):
        parent_client = self.var('parentClient')

        self.elements = self.create_children(Element, self._get_elements(parent_client))

        opts = self.var('options')
        if not opts and parent_client:
            opts = gws.get(parent_client, 'options', {})
        self.options = opts or {}

    def _get_elements(self, parent_client):
        elements = self.var('elements')
        if elements:
            return elements

        if not parent_client:
            return []

        elements = list(gws.get(parent_client, 'elements', []))

        add = self.var('addElements', default=[])

        for c in add:
            n = _find_element(elements, c.tag)
            if n >= 0:
                elements.pop(n)
            if c.before:
                n = _find_element(elements, c.before)
                if n >= 0:
                    elements.insert(n, c)
            elif c.after:
                n = _find_element(elements, c.after)
                if n >= 0:
                    elements.insert(n + 1, c)
            else:
                elements.append(c)

        remove = self.var('removeElements', default=[])
        remove_tags = [c.tag for c in remove]
        return [e for e in elements if e.tag not in remove_tags]


def _find_element(elements, tag):
    for n, el in enumerate(elements):
        if el.tag == tag:
            return n
    return -1
