import os
import re

import gws
import gws.types as t
import gws.lib.net
import gws.lib.os2

EXEC_PATH = '/usr/bin/qgis_mapserv.fcgi'
SVG_SEARCH_PATHS = ['/usr/share/qgis/svg', '/usr/share/alkisplugin/svg']


def _make_ini(root, base_dir):
    ini = ''

    paths = []
    s = root.application.var('server.qgis.searchPathsForSVG')
    if s:
        paths.extend(s)
    paths.extend(SVG_SEARCH_PATHS)
    ini += f'''
        [svg]
        searchPathsForSVG={','.join(paths)}        
    '''

    # set the cache dir and size=4096Kb
    gws.ensure_dir('netcache', base_dir)
    ini += fr'''
        [cache]
        directory={base_dir}/netcache
        size=@Variant(\0\0\0\x81\0\0\0\0\0@\0\0)
    '''

    proxy = os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY')
    if proxy:
        p = gws.lib.net.parse_url(proxy)
        ini += f'''
            [proxy]
            proxyEnabled=true
            proxyType=HttpProxy
            proxyHost={p.hostname}
            proxyPort={p.port}
            proxyUser={p.username}
            proxyPassword={p.password}
        '''

    return '\n'.join(x.strip() for x in ini.splitlines())


def environ(root: gws.RootObject):
    base_dir = gws.ensure_dir(gws.TMP_DIR + '/qqq')

    # it's all a bit blurry, but the server appears to read 'ini' from OPTIONS_DIR
    # while the app uses a profile
    # NB: for some reason, the profile path will be profiles/profiles/default (sic!)

    gws.ensure_dir('profiles', base_dir)
    gws.ensure_dir('profiles/default', base_dir)
    gws.ensure_dir('profiles/default/QGIS', base_dir)
    gws.ensure_dir('profiles/profiles', base_dir)
    gws.ensure_dir('profiles/profiles/default', base_dir)
    gws.ensure_dir('profiles/profiles/default/QGIS', base_dir)

    ini = _make_ini(root, base_dir)
    gws.write_file(base_dir + '/profiles/default/QGIS/QGIS3.ini', ini)
    gws.write_file(base_dir + '/profiles/profiles/default/QGIS/QGIS3.ini', ini)

    # server options, as documented on
    # see https://docs.qgis.org/testing/en/docs/user_manual/working_with_ogc/server/config.html#environment-variables

    server_env = {
        # not used here 'QGIS_PLUGINPATH': '',
        # not used here 'QGIS_SERVER_LOG_FILE': '',

        # see https://github.com/qgis/QGIS/pull/35738
        'QGIS_SERVER_IGNORE_BAD_LAYERS': 'true',

        'MAX_CACHE_LAYERS': root.application.var('server.qgis.maxCacheLayers'),
        'QGIS_OPTIONS_PATH': base_dir + '/profiles/profiles/default',
        'QGIS_SERVER_CACHE_DIRECTORY': gws.ensure_dir('servercache', base_dir),
        'QGIS_SERVER_CACHE_SIZE': root.application.var('server.qgis.serverCacheSize'),
        'QGIS_SERVER_LOG_LEVEL': root.application.var('server.qgis.serverLogLevel'),
        # 'QGIS_SERVER_MAX_THREADS': 4,
        # 'QGIS_SERVER_PARALLEL_RENDERING': 'false',
    }

    # qgis app options, mostly undocumented

    qgis_env = {
        'QGIS_PREFIX_PATH': '/usr',
        'QGIS_DEBUG': root.application.var('server.qgis.debug'),
        # 'QGIS_GLOBAL_SETTINGS_FILE': '/global_settings.ini',
        'QGIS_CUSTOM_CONFIG_PATH': base_dir
    }

    # finally, there are lots of GDAL settings, some of those seem relevant
    # http://trac.osgeo.org/gdal/wiki/ConfigOptions

    gdal_env = {
        'GDAL_FIX_ESRI_WKT': 'GEOGCS',
        'GDAL_DEFAULT_WMS_CACHE_PATH': gws.ensure_dir('gdalcache', base_dir),
    }

    return gws.merge(
        server_env,
        qgis_env,
        gdal_env,
    )


def version():
    _, txt = gws.lib.os2.run([EXEC_PATH])
    m = re.search(r'QGis version (.+)', gws.as_str(txt))
    if m:
        return m.group(1).strip()
    return 'unknown'


def request(root: gws.RootObject, params, **kwargs):
    """Make a request to the local qgis server"""

    url = 'http://%s:%s' % (
        root.application.var('server.qgis.host'),
        root.application.var('server.qgis.port'))

    return gws.lib.net.http_request(url, params=params, **kwargs)
