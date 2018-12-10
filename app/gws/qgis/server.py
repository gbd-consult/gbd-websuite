import os
import re

import gws
import gws.config
import gws.tools.misc as misc
import gws.tools.shell as sh
import gws.tools.net

EXEC_PATH = '/usr/bin/qgis_mapserv.fcgi'
SVG_SEARCH_PATHS = ['/usr/share/qgis/svg', '/usr/share/alkisplugin/svg']


def _make_ini(cfg):
    ini = ''

    paths = []
    if cfg.searchPathsForSVG:
        paths.extend(cfg.searchPathsForSVG)
    paths.extend(SVG_SEARCH_PATHS)
    ini += f'''
        [svg]
        searchPathsForSVG={','.join(paths)}        
    '''

    proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
    if proxy:
        p = gws.tools.net.parse_url(proxy)
        ini += f'''
            [proxy]
            proxyEnabled=true
            proxyType=HttpProxy
            proxyHost={p['hostname']}
            proxyPort={p['port']}
            proxyUser={p['username']}
            proxyPassword={p['password']}
        '''

    return '\n'.join(x.strip() for x in ini.splitlines())


def environ(cfg):
    base_dir = misc.ensure_dir('/tmp/qqq')

    # it's all a bit blurry, but the server appears to read 'ini' from OPTIONS_DIR
    # while the app uses a profile
    # NB: for some reason, the profile path will be profiles/profiles/default (sic!)
    # ok, let's the make them point to the same file

    profile_dir = misc.ensure_dir('profiles/profiles/default', base_dir)
    ini_dir = misc.ensure_dir('QGIS', profile_dir)

    ini = _make_ini(cfg)
    with open(ini_dir + '/QGIS3.ini', 'wt') as fp:
        fp.write(ini)

    # server options, as documented on
    # see https://docs.qgis.org/testing/en/docs/user_manual/working_with_ogc/server/config.html#environment-variables

    server_env = {
        # not used here 'QGIS_PLUGINPATH': '',
        # not used here 'QGIS_SERVER_LOG_FILE': '',
        'MAX_CACHE_LAYERS': cfg.maxCacheLayers,
        'QGIS_OPTIONS_PATH': profile_dir,
        'QGIS_SERVER_CACHE_DIRECTORY': misc.ensure_dir('servercache', base_dir),
        'QGIS_SERVER_CACHE_SIZE': cfg.serverCacheSize,
        'QGIS_SERVER_LOG_LEVEL': cfg.serverLogLevel,
        # 'QGIS_SERVER_MAX_THREADS': 4,
        # 'QGIS_SERVER_PARALLEL_RENDERING': 'false',
    }

    # qgis app options, mostly undocumented

    qgis_env = {
        'QGIS_PREFIX_PATH': '/usr',
        'QGIS_DEBUG': cfg.debug,
        # 'QGIS_GLOBAL_SETTINGS_FILE': '/global_settings.ini',
        'QGIS_CUSTOM_CONFIG_PATH': base_dir
    }

    # finally, there are lots of GDAL settings, some of those seem relevant
    # http://trac.osgeo.org/gdal/wiki/ConfigOptions

    gdal_env = {
        'GDAL_FIX_ESRI_WKT': 'GEOGCS',
        'GDAL_DEFAULT_WMS_CACHE_PATH': misc.ensure_dir('gdalcache', base_dir),
    }

    return gws.extend(
        server_env,
        qgis_env,
        gdal_env,
    )


def version():
    _, txt = sh.run([EXEC_PATH])
    m = re.search(r'QGis version (.+)', gws.as_str(txt))
    if m:
        return m.group(1).strip()
    return 'unknown'
