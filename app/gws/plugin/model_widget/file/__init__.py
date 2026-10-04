"""File widget.

Upload and download of the file in a ``file`` field. This is the default
widget of ``file`` fields.

Example::

    fields+ {
        name "photo"
        type "file"
        contentColumn "photo_content"
        nameColumn "photo_name"
        widget.type "file"
    }
"""

import gws
import gws.base.model.widget


@gws.ext.config.modelWidget('file')
class Config(gws.base.model.widget.Config):
    """Upload and download for a file field."""

    pass


@gws.ext.props.modelWidget('file')
class Props(gws.base.model.widget.Props):
    pass


@gws.ext.object.modelWidget('file')
class Object(gws.base.model.widget.Object):
    """File widget object."""

    pass
