"""Alias layer."""

import gws.config
import gws.common.layer
import gws.gis.extent
import gws.gis.legend

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Alias layer"""

    target: str


class Object(gws.gws.Object):
    target_obj = None
    root = None

    def configure(self):
        super().configure()
        self.map: t.IMap = t.cast(t.IMap, self.get_closest('gws.common.map'))

        uid = self.var('uid') or gws.as_uid(self.title) or 'layer'
        if self.map:
            uid = self.map.uid + '.' + uid
        self.set_uid(uid)

    def post_configure(self):
        pass

    def activate(self):
        self.target_obj = self.root.find_by_uid(self.var('target'))

    @property
    def props(self):
        ta = self._get_target()
        if ta:
            p = ta.props
            if p.url:
                p.url = p.url.replace(p.uid, self.uid)
            p.uid = self.uid
            return p
        return t.Data()

    def __getattr__(self, item):
        ta = self._get_target()
        if ta:
            v = getattr(ta, item)
            return v
        raise AttributeError(f'no target found for {item!r}')

    def _get_target(self):
        if not self.target_obj and self.root:
            self.target_obj = self.root.find_by_uid(self.var('target'))
        return self.target_obj
