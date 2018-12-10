import re

import gws
import gws.config
import gws.types as t

from . import error

_no_site = t.Data({
    'host': '*'
})


def _site(environ):
    h = environ.get('HTTP_HOST', '').lower().split(':')[0].strip()

    sites = gws.config.var('web.sites')
    if not sites:
        return _no_site

    for s in sites:
        if s.host.lower() == h:
            return s
    for s in sites:
        if s.host == '*':
            return s

    gws.log.error('unknown host', h)
    raise error.NotFound()


def _rewrite(url, rules):
    for rule in rules:
        m = re.match(rule.match, url)
        if m:
            params = {}
            for k, v in rule.route.params.items():
                params[k] = re.sub(r'\$(\d+)', lambda x: m.group(int(x.group(1))), v)
            route = {
                'cmd': rule.route.cmd,
                'params': params
            }
            gws.log.info('rewrite', url, '=>', route)
            return route


def prepare(environ):
    site = environ['gws.site'] = _site(environ)
    rm = environ.get('REQUEST_METHOD', '').upper()

    # if rm in ('GET', 'HEAD') and site.get('rewrite'):
    #     route = _rewrite(environ.get('REQUEST_URI'), site.rewrite)
    #     if route:
    #         environ['gws.rewrite_route'] = route
    #         environ['PATH_INFO'] = gws.SERVER_ENDPOINT
    #
    return environ
