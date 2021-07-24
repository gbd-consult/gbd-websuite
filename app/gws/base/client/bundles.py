"""Deal with client bundles created by the js bundler (app/js/helpers/builder.js)"""

import gws
import gws.lib.os2
import gws.lib.json2

BUNDLE_KEY_TEMPLATE = 'TEMPLATE'
BUNDLE_KEY_MODULES = 'MODULES'
BUNDLE_KEY_STRINGS = 'STRINGS'
BUNDLE_KEY_CSS = 'CSS'

DEFAULT_LANG = 'de'
DEFAULT_THEME = 'light'


def load(root: gws.RootObject):
    def _load():
        bundles = {}

        for path in root.specs.client_bundle_paths():
            gws.log.info(f'loading bundle {path!r}')
            bundle = gws.lib.json2.from_path(path)
            for key, val in bundle.items():
                if key not in bundles:
                    bundles[key] = ''
                bundles[key] += val

        return bundles

    if root.application.developer_option('reload_bundles'):
        return _load()

    return gws.get_server_global('CLIENT_BUNDLES', _load)


def javascript(root: gws.RootObject, locale_uid: str):
    bundles = load(root)
    lang = locale_uid.split('_')[0]
    modules = bundles[BUNDLE_KEY_MODULES]
    strings = bundles.get(BUNDLE_KEY_STRINGS + '_' + lang) or bundles.get(BUNDLE_KEY_STRINGS + '_' + DEFAULT_LANG)

    js = bundles[BUNDLE_KEY_TEMPLATE]
    js = js.replace('__MODULES__', modules)
    js = js.replace('__STRINGS__', strings)

    return js


def vendor_javascript(root: gws.RootObject):
    return gws.read_file(root.specs.client_vendor_bundle_path())


def css(root: gws.RootObject, theme: str):
    bundles = load(root)
    return bundles.get(BUNDLE_KEY_CSS + '_' + theme) or bundles.get(BUNDLE_KEY_CSS + '_' + DEFAULT_THEME)
