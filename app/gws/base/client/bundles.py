"""Deal with client bundles created by the js bundler (app/js/helpers/builder.js)"""

import gws
import gws.lib.json2

BUNDLE_KEY_TEMPLATE = 'TEMPLATE'
BUNDLE_KEY_MODULES = 'MODULES'
BUNDLE_KEY_STRINGS = 'STRINGS'
BUNDLE_KEY_CSS = 'CSS'

DEFAULT_LANG = 'de'
DEFAULT_THEME = 'light'


def javascript(root: gws.IRoot, category: str, locale_uid: str = '') -> str:
    if category == 'vendor':
        return gws.read_file(root.specs.bundle_paths('vendor')[0])

    if category == 'util':
        return gws.read_file(root.specs.bundle_paths('util')[0])

    if category == 'app':
        bundles = _load_app_bundles(root)
        lang = locale_uid.split('_')[0]
        modules = bundles[BUNDLE_KEY_MODULES]
        strings = bundles.get(BUNDLE_KEY_STRINGS + '_' + lang) or bundles.get(BUNDLE_KEY_STRINGS + '_' + DEFAULT_LANG)

        js = bundles[BUNDLE_KEY_TEMPLATE]
        js = js.replace('__MODULES__', modules)
        js = js.replace('__STRINGS__', strings)

        return js


def css(root: gws.IRoot, category: str, theme: str):
    if category == 'app':
        bundles = _load_app_bundles(root)
        return bundles.get(BUNDLE_KEY_CSS + '_' + theme) or bundles.get(BUNDLE_KEY_CSS + '_' + DEFAULT_THEME)
    return ''


##

def _load_app_bundles(root: gws.IRoot):
    def _load():
        bundles = {}

        for path in root.specs.bundle_paths('app'):
            gws.log.info(f'loading bundle {path!r}')
            bundle = gws.lib.json2.from_path(path)
            for key, val in bundle.items():
                if key not in bundles:
                    bundles[key] = ''
                bundles[key] += val

        return bundles

    if root.application.developer_option('web.reload_bundles'):
        return _load()

    return gws.get_server_global('APP_BUNDLES', _load)
