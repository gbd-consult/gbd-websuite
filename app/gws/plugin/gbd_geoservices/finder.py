"""GBD Geoservices finder."""

import os

import gws
import gws.base.search
import gws.config.util


gws.ext.new.finder('gbd_geoservices')

_DEFAULT_TEMPLATES = [
    gws.Config(
        subject='feature.teaser',
        type='html',
        path=os.path.dirname(__file__) + '/templates/feature_teaser.cx.html',
    ),
    gws.Config(
        subject='feature.description',
        type='html',
        path=os.path.dirname(__file__) + '/templates/feature_description.cx.html',
    ),
]


class Config(gws.base.search.finder.Config):
    """GBD Geoservices search"""

    apiKey: str
    """API key."""


class Object(gws.base.search.finder.Object):
    supportsKeywordSearch = True
    supportsGeometrySearch = True

    def configure(self):
        self.configure_models()
        self.configure_templates()

    def configure_templates(self):
        return gws.config.util.configure_templates_for(self, extra=_DEFAULT_TEMPLATES)

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            apiKey=self.cfg('apiKey'),
        )
