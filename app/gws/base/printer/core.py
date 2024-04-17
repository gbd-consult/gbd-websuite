from typing import Optional

import gws
import gws.base.feature
import gws.base.model
import gws.base.template
import gws.config.util
import gws.lib.job
import gws.lib.style

gws.ext.new.printer('default')


class Config(gws.Config):
    """Printer configuration"""

    template: gws.ext.config.template
    """Print template"""
    title: str = ''
    """Printer title"""
    models: Optional[list[gws.ext.config.model]]
    """Data models"""
    qualityLevels: Optional[list[gws.TemplateQualityLevel]]
    """Quality levels supported by this printer"""


class Props(gws.Props):
    template: gws.base.template.Props
    model: gws.base.model.Props
    qualityLevels: list[gws.TemplateQualityLevel]
    title: str


class Object(gws.Printer):

    def configure(self):
        gws.config.util.configure_models(self)
        self.template = self.create_child(gws.ext.object.template, self.cfg('template'))
        self.qualityLevels = self.cfg('qualityLevels') or [gws.TemplateQualityLevel(name='default', dpi=0)]
        self.title = self.cfg('title') or self.template.title or ''

    def props(self, user):
        model = self.root.app.modelMgr.locate_model(self, user=user, access=gws.Access.write)
        return Props(
            uid=self.uid,
            template=self.template,
            model=model,
            qualityLevels=self.qualityLevels,
            title=self.title,
        )
