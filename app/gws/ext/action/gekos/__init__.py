import time

import gws
import gws.auth.api
import gws.web
import gws.config
import gws.tools.net
import gws.gis.feature
import gws.gis.layer

import gws.types as t


class Config(t.WithTypeAndAccess):
    """gekos action"""

    url: t.url
    crs: t.crsref
    params: dict
    allInstances: t.Optional[t.List[str]]
    table: str = ''


class Object(gws.Object):
    def configure(self):
        super().configure()
        self.crs = self.var('crs', parent=True)

    def http_get_xy_from_fs(self, req, p) -> t.HttpResponse:
        # ?cmd=gekosHttpGetXyFromFs&gemarkungUid=<gem>&flurnummer=<flur>&zaehler=<zaehler>&nenner=<nenner>&flurstuecksfolge=<folge>
        # httpResponse mode, s. spec page 19

        query = gws.pick(req.params, [
            'gemarkungUid',
            'flurnummer',
            'zaehler',
            'nenner',
            'flurstuecksfolge',
        ])

        # '0' values should be NULLs
        query = {k: None if v == '0' else v for k, v in query.items()}

        gws.p('http_get_xy_from_fs', req.params)

        if not query:
            gws.log.warn('gekos: bad request')
            return _text('error:')

        alkis = self.root.find_first('gws.ext.action.alkis')
        total, features = alkis.find_fs(t.Data(query), self.crs, limit=1)

        if total == 0:
            gws.log.warn('gekos: not found')
            return _text('error:')

        return _text_xy(features[0])

    def http_get_xy_from_grd(self, req, p) -> t.HttpResponse:
        # ?cmd=gekosHttpGetXyFromFs&gemarkungUid=<gem>&flurnummer=<flur>&zaehler=<zaehler>&nenner=<nenner>&flurstuecksfolge=<folge>
        # httpResponse mode, s. spec page 19

        query = gws.pick(req.params, [
            'gemeinde',
            'strasse',
            'hausnummer',
        ])

        if not query:
            gws.log.warn('gekos: bad request')
            return _text('error:')

        # if no hnr is given, only select locations that have one
        if 'hausnummer' not in query or query['hausnummer'] == '0':
            query['hausnummer'] = '*'

        alkis = self.root.find_first('gws.ext.action.alkis')
        total, features = alkis.find_address(t.Data(query), self.crs, limit=1)

        if total == 0:
            gws.log.warn('gekos: not found')
            return _text('error:')

        return _text_xy(features[0])


def _text(s):
    return t.HttpResponse({
        'mimeType': 'text/plain',
        'content': s
    })


def _text_xy(f):
    return _text('%.3f;%.3f' % (f.attributes['x'], f.attributes['y']))
