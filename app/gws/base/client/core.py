"""Client object and UI elements."""

from typing import Optional

import gws


class ElementConfig(gws.ConfigWithAccess):
    """Client UI element."""

    tag: str
    """Element tag."""
    before: str = ''
    """Tag of the element to insert this one before."""
    after: str = ''
    """Tag of the element to insert this one after."""
    options: Optional[dict]
    """Element-specific options passed to the client."""


class Config(gws.ConfigWithAccess):
    """UI elements and options of the browser client."""

    options: Optional[dict]
    """Client options, merged with the application client options."""
    elements: Optional[list[ElementConfig]]
    """Client UI elements, replacing the inherited application list."""
    addElements: Optional[list[ElementConfig]]
    """Elements to add to the inherited application element list."""
    removeElements: Optional[list[ElementConfig]]
    """Elements to remove from the inherited application element list."""


class ElementProps(gws.Data):
    """Client UI element properties."""

    tag: str


class Props(gws.Data):
    """Client properties."""

    options: Optional[dict]
    elements: Optional[list[ElementProps]]


class Element(gws.Node):
    """Client UI element.

    An element of the browser client, identified by its tag, with element options
    and access rules.
    """

    tag: str
    """Element tag, for example ``Toolbar.Print``."""
    after: str
    """Tag of the element this one was inserted after."""
    before: str
    """Tag of the element this one was inserted before."""
    options: dict
    """Element-specific options."""

    def configure(self):
        self.tag = self.cfg('tag')
        self.after = self.cfg('after')
        self.before = self.cfg('before')
        self.options = self.cfg('options') or {}

    def props(self, user):
        return gws.Data(tag=self.tag, options=self.options)


class Object(gws.Client):
    """Client object."""

    options: dict
    """Client options."""
    elements: list[Element]
    """Client UI elements."""

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
        """Return the element list, either configured or inherited from the application client and modified.

        An added element replaces an inherited one with the same tag. An element
        with ``before`` or ``after`` is only added if the referenced tag is present.
        """
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
        """Return the index of the element with the given tag, or -1."""
        for n, el in enumerate(elements):
            if el.tag == tag:
                return n
        return -1
