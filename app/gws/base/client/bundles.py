"""Deal with client bundles created by the js bundler (app/js/helpers/builder.js)"""

import gws
import gws.lib.jsonx

BUNDLE_KEY_TEMPLATE = 'TEMPLATE'
BUNDLE_KEY_MODULES = 'MODULES'
BUNDLE_KEY_STRINGS = 'STRINGS'
BUNDLE_KEY_CSS = 'CSS'

DEFAULT_LANG = 'de'
DEFAULT_THEME = 'light'


def javascript(root: gws.IRoot, category: str, locale_uid: str = '') -> str:
    if category == 'vendor':
        return gws.read_file(gws.APP_DIR + '/' + gws.JS_VENDOR_BUNDLE)

    if category == 'util':
        return gws.read_file(gws.APP_DIR + '/' + gws.JS_UTIL_BUNDLE)

    if category == 'app':
        return _make_app_js(root, locale_uid)


def css(root: gws.IRoot, category: str, theme: str):
    if category == 'app':
        bundles = _load_app_bundles(root)
        return bundles.get(BUNDLE_KEY_CSS + '_' + theme) or bundles.get(BUNDLE_KEY_CSS + '_' + DEFAULT_THEME)
    return ''


##

def _load_app_bundles(root):

    def _load():
        bundles = {}

        for path in root.specs.appBundlePaths:
            if gws.is_file(path):
                gws.log.debug(f'bundle {path!r}: loading')
                bundle = gws.lib.jsonx.from_path(path)
                for key, val in bundle.items():
                    if key not in bundles:
                        bundles[key] = ''
                    bundles[key] += val

        return bundles

    if root.app.developer_option('web.reload_bundles'):
        return _load()

    return gws.get_server_global('APP_BUNDLES', _load)


def _make_app_js(root, locale_uid):
    bundles = _load_app_bundles(root)

    lang = locale_uid.split('_')[0]
    modules = bundles[BUNDLE_KEY_MODULES]
    strings = bundles.get(BUNDLE_KEY_STRINGS + '_' + lang) or bundles.get(BUNDLE_KEY_STRINGS + '_' + DEFAULT_LANG)

    js = bundles[BUNDLE_KEY_TEMPLATE]
    js = js.replace('__MODULES__', modules)
    js = js.replace('__STRINGS__', strings)

    return js
