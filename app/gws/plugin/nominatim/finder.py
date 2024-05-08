"""Nominatim finder.

http://wiki.openstreetmap.org/wiki/Nominatim
https://nominatim.org/release-docs/develop/api/Search/

"""

from typing import Optional

import gws
import gws.base.search
import gws.base.template
import gws.config.util
import gws.lib.net


gws.ext.new.finder('nominatim')

_DEFAULT_TEMPLATES = [
    gws.Config(
        subject='feature.teaser',
        type='html',
        text='''
            <p class="head">{name|html}</p>
            <p>{osm_class}: {osm_type}</p>
        '''
    ),
    gws.Config(
        subject='feature.description',
        type='html',
        text='''
            <p class="head">{name|html}</p>
            <p class="head2">{osm_class}: {osm_type}</p>
            <p class="text2">{address_road} {address_building}
                <br>{address_postcode} {address_city}
                <br>{address_country}
            </p>
        '''
    ),
]


class Config(gws.base.search.finder.Config):
    """Nominatim search"""

    country: Optional[str]
    """country to limit the search"""
    language: Optional[str]
    """language to return the results in"""


class Object(gws.base.search.finder.Object):
    supportsKeywordSearch = True

    def configure(self):
        self.configure_models()
        self.configure_templates()

    def configure_templates(self):
        return gws.config.util.configure_templates(self, extra=_DEFAULT_TEMPLATES)

    def configure_models(self):
        return gws.config.util.configure_models(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            country=self.cfg('country'),
            language=self.cfg('language')
        )
