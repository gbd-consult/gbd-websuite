import gws
import gws.lib.style
from gws.lib.style import Record, Props
import gws.types as t


class Config(gws.Config):
    """Feature style"""

    name: t.Optional[str]  #: style name
    selector: t.Optional[str]  #: CSS selector
    text: t.Optional[str]  #: raw style content
    values: t.Optional[dict]  #: style values


class Object(gws.Object, gws.IStyle):
    _rec: Record

    @property
    def props(self):
        return Props(
            values=vars(self._rec.values),
            text=self._rec.text or '',
            name=self._rec.name or '',
            selector=self._rec.selector or '',
        )

    def configure(self):
        self._rec = gws.lib.style.from_dict(gws.as_dict(self.config))
