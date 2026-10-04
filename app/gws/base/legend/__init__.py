"""Base classes for layer legends.

A legend belongs to a layer and renders an image that explains the layer's
symbols. The base package provides the generic ``Object`` with the common
``Config`` options (``cacheMaxAge``, ``options``) and ``combine_outputs``, which
stacks several legend images vertically into one. Concrete legend types are
plugins in ``gws.plugin.legend`` (``static``, ``remote``, ``html``,
``combined``) and in other plugins such as ``gws.plugin.qgis``. A group layer
without its own legend gets a ``combined`` legend of its children.

Layers render their legends with ``Layer.render_legend``, the client fetches
them with the ``mapGetLegend`` command.

Submodules:

- ``core``: the base ``Object``, ``Config``, ``Props`` and ``combine_outputs``.

Example::

    map.layers+ {
        type "qgis"
        provider.path "/data/projects/roads.qgs"
        legend { type "static" path "/data/legends/roads.png" }
    }

A legend type in a plugin::

    @gws.ext.object.legend('mylegend')
    class Object(gws.base.legend.Object):
        def render(self, args=None):
            images = [gws.lib.image.from_path(p) for p in self.cfg('paths')]
            lros = [gws.LegendRenderOutput(image=img, size=img.size()) for img in images]
            return gws.base.legend.combine_outputs(lros, self.options)
"""

from .core import (
    Config,
    Object,
    Props,
    combine_outputs,
)
