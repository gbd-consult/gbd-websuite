"""Model manager."""

import gws
import gws.lib.mime


class Object(gws.TemplateManager):
    def find_template(self, *objects, user=None, subject=None, mime=None):
        mime = gws.lib.mime.get(mime) if mime else None

        for obj in objects:
            if not obj:
                continue
            p = self._find(obj, user, subject, mime)
            if p:
                gws.log.debug(f'find_template: found {subject=} {obj=}')
                return p

        p = self._find(self.root.app, user, subject, mime)
        if p:
            gws.log.debug(f'find_template: found {subject=} APP')
            return p

    def _find(self, obj, user, subject, mime):
        for tpl in getattr(obj, 'templates', []):
            if subject and tpl.subject != subject:
                continue
            if user and not user.can_use(tpl):
                continue
            if mime and tpl.mimeTypes and mime not in tpl.mimeTypes:
                continue
            return tpl

    def template_from_path(self, path, **kwargs):
        for ext, typ in _TYPES.items():
            if path.endswith(ext):
                return self.root.create_shared(
                    gws.ext.object.template,
                    gws.Config(uid=gws.u.sha256(path), type=typ, path=path),
                    **kwargs
                )


# @TODO template types should be configurable

_TYPES = {
    '.cx.html': 'html',
    '.cx.csv': 'csv',
    '.qgs': 'qgis',
    '.cx.xml': 'xml',
}
