"""Client JavaScript and CSS bundles created by the JS bundler."""

import gws
import gws.lib.jsonx

BUNDLE_KEY_TEMPLATE = 'TEMPLATE'
BUNDLE_KEY_MODULES = 'MODULES'
BUNDLE_KEY_STRINGS = 'STRINGS'
BUNDLE_KEY_CSS = 'CSS'

DEFAULT_LANG = 'de'
DEFAULT_THEME = 'light'


def javascript(root: gws.Root, category: str, locale: gws.Locale) -> str:
    """Return a JavaScript bundle.

    Args:
        root: The configuration root.
        category: ``vendor`` for the vendor libraries, ``app`` for the client
            application with the UI strings for the locale language
            (German if not available).
        locale: The client locale.

    Returns:
        The JavaScript code, or ``None`` for an unknown category.
    """
    if category == 'vendor':
        return gws.u.read_file(gws.c.APP_DIR + '/' + gws.c.JS_VENDOR_BUNDLE)

    if category == 'app':
        return _make_app_js(root, locale)


def css(root: gws.Root, category: str, theme: str):
    """Return a CSS bundle.

    Args:
        root: The configuration root.
        category: Bundle category, only ``app`` has CSS.
        theme: Theme name, ``light`` if empty.

    Returns:
        The CSS code, ``None`` if the theme is not found, or an empty string for other categories.
    """
    if category == 'app':
        bundles = _load_app_bundles(root)
        theme = theme or DEFAULT_THEME
        return bundles.get(BUNDLE_KEY_CSS + '_' + theme)
    return ''


##

def _load_app_bundles(root):
    """Load and concatenate the application bundle files, cached per server unless the ``web.reload_bundles`` developer option is set."""

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
    """Build the application script from the bundle template, the modules and the UI strings."""
    bundles = _load_app_bundles(root)

    modules = bundles[BUNDLE_KEY_MODULES]
    strings = bundles.get(BUNDLE_KEY_STRINGS + '_' + locale.language) or bundles.get(BUNDLE_KEY_STRINGS + '_' + DEFAULT_LANG)

    js = bundles[BUNDLE_KEY_TEMPLATE]
    js = js.replace('__MODULES__', modules)
    js = js.replace('__STRINGS__', strings)

    return js
