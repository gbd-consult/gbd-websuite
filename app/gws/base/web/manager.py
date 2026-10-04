"""Web manager."""

from typing import Optional

import gws

from . import site

class Config(gws.Config):
    """Web server settings."""

    site: Optional[site.Config]
    """Web site settings."""
    sites: Optional[list[site.Config]]
    """List of sites. (deprecated in 8.4)"""
    ssl: Optional[site.SSLConfig]
    """SSL settings."""


class Object(gws.WebManager):
    """Web manager."""

    def configure(self):
        p = self.cfg('site')
        if not p:
            # deprecated
            cfgs = self.cfg('sites') or []
            if cfgs:
                self.root.config_warning('"web.sites" is deprecated, use "web.site"')
            if len(cfgs) > 1:
                raise gws.ConfigurationError('multiple web sites are not supported')
            p = cfgs[0] if cfgs else gws.Config()
        if self.cfg('ssl'):
            p = gws.u.merge(p, ssl=True)
        self.site = self.create_child(site.Object, p)

        # deprecated
        self.sites = [self.site]
