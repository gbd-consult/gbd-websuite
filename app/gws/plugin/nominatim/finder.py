"""Nominatim finder.

http://wiki.openstreetmap.org/wiki/Nominatim
https://nominatim.org/release-docs/develop/api/Search/

"""

import gws
import gws.types as t
import gws.base.search
import gws.base.template
import gws.lib.net

gws.ext.new.finder('nominatim')

_DEFAULT_TEMPLATES = [
    gws.Config(
        subject='feature.teaser',
        type='html',
        text='''
            <p class="head">{name|html}</p>
            <p>{osm_class}, {osm_type}</p>
        '''
    ),
    gws.Config(
        subject='feature.description',
        type='html',
        text='''
            <p class="head">{name|html}</p>
            <p class="head2">{osm_class}, {osm_type}</p>
            <p class="text2">{address_road} {address_building}
                <br>{address_postcode} {address_city}
                <br>{address_country}
            </p>
        '''
    ),
]


class Config(gws.base.search.finder.Config):
    """Nominatim search"""

    country: t.Optional[str]
    """country to limit the search"""
    language: t.Optional[str]
    """language to return the results in"""


class Object(gws.base.search.finder.Object):
    supportsKeywordSearch = True

    def configure(self):
        self.configure_models()
        self.configure_templates()

    def configure_templates(self):
        if super().configure_templates():
            return True
        self.templates = [self.configure_template(cfg) for cfg in _DEFAULT_TEMPLATES]
        return True

    def configure_models(self):
        if super().configure_models():
            return True
        self.models.append(self.configure_model(None))
        return True

    def configure_model(self, cfg):
        return self.create_child(gws.ext.object.model, cfg, type=self.extType, country=self.cfg('country'), language=self.cfg('language'))
