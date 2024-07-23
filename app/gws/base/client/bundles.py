"""Deal with client bundles created by the js bundler (app/js/helpers/builder.js)"""

import gws
import gws.lib.jsonx

BUNDLE_KEY_TEMPLATE = 'TEMPLATE'
BUNDLE_KEY_MODULES = 'MODULES'
BUNDLE_KEY_STRINGS = 'STRINGS'
BUNDLE_KEY_CSS = 'CSS'

DEFAULT_LANG = 'de'
DEFAULT_THEME = 'light'


def javascript(root: gws.Root, category: str, locale: gws.Locale) -> str:
    if category == 'vendor':
        return gws.u.read_file(gws.c.APP_DIR + '/' + gws.c.JS_VENDOR_BUNDLE)

    if category == 'util':
        return gws.u.read_file(gws.c.APP_DIR + '/' + gws.c.JS_UTIL_BUNDLE)

    if category == 'app':
        return _make_app_js(root, locale)


def css(root: gws.Root, category: str, theme: str):
    if category == 'app':
        bundles = _load_app_bundles(root)
        theme = theme or DEFAULT_THEME
        return bundles.get(BUNDLE_KEY_CSS + '_' + theme)
    return ''


##

def _load_app_bundles(root):

    def _load():
        bundles = {}

        for path in root.specs.appBundlePaths:
            if gws.u.is_file(path):
                gws.log.debug(f'bundle {path!r}: loading')
                bundle = gws.lib.jsonx.from_path(path)
                for key, val in bundle.items():
                    if key not in bundles:
                        bundles[key] = ''
                    bundles[key] += val

        return bundles

    if root.app.developer_option('web.reload_bundles'):
        return _load()

    return gws.u.get_server_global('APP_BUNDLES', _load)


def _make_app_js(root, locale):
    bundles = _load_app_bundles(root)

    modules = bundles[BUNDLE_KEY_MODULES]
    strings = bundles.get(BUNDLE_KEY_STRINGS + '_' + locale.language) or bundles.get(BUNDLE_KEY_STRINGS + '_' + DEFAULT_LANG)

    js = bundles[BUNDLE_KEY_TEMPLATE]
    js = js.replace('__MODULES__', modules)
    js = js.replace('__STRINGS__', strings)

    return js
