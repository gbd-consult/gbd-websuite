"""XML helper object.

Used in the configuration to configure XML options and additional namespaces.
"""

from typing import Optional

import gws

from . import namespace

gws.ext.new.helper('xml')


class Config(gws.Config):
    """XML settings"""

    namespaces: Optional[list[gws.XmlNamespace]]
    """custom XML namespaces"""


class Object(gws.Node):
    def activate(self):
        p = self.cfg('namespaces')
        if p:
            for ns in p:
                namespace.register(ns)
