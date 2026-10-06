"""Template manager."""

import gws
import gws.lib.mime

# @TODO template types should be configurable

TEMPLATE_TYPES = {
    '.cx.html': 'html',
    '.qgs': 'qgis',
    '.cx.py': 'py',
}


class Object(gws.TemplateManager):
    """Template manager."""

    def find_templates(self, subjects, where, user=None, mime_type=None):
        return gws.u.compact(self.find_template(s, where, user, mime_type) for s in subjects)

    def find_template(self, subject, where, user=None, mime_type=None):
        for obj in where:
            if not obj:
                continue
            p = self._find(subject, obj, user, mime_type)
            if p:
                gws.log.debug(f'find_template: found {subject=} {obj=}')
                return p

        p = self._find(subject, self.root.app, user, mime_type)
        if p:
            gws.log.debug(f'find_template: found {subject=} APP')
            return p

    def _find(self, subject, obj, user, mime_type):
        """Return the first matching template of an object."""
        for tpl in getattr(obj, 'templates', []):
            if tpl.subject != subject:
                continue
            if user and not user.can_use(tpl):
                continue
            if mime_type and tpl.mimeTypes and mime_type not in tpl.mimeTypes:
                continue
            return tpl

    def template_from_path(self, path, **kwargs):
        for ext, typ in TEMPLATE_TYPES.items():
            if path.endswith(ext):
                return self.root.create_shared(
                    gws.ext.object.template,
                    gws.Config(uid=gws.u.sha256(path), type=typ, path=path),
                    **kwargs
                )
