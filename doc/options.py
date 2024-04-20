import os

SELF_DIR = os.path.dirname(__file__)

ROOT_DIR = os.path.realpath(SELF_DIR + '/../')
DOC_DIR = SELF_DIR
APP_DIR = ROOT_DIR + '/app'
BUILD_DIR = APP_DIR + '/__build'

VERSION3 = open(APP_DIR + '/VERSION').read().strip()
VERSION2 = VERSION3.rpartition('.')[0]

# dog options (see lib/vendor/dog/options.py)

docRoots = [ROOT_DIR]
docPatterns = ['*.doc.md']
assetPatterns = ['*.svg', '*.png']
excludeRegex = 'node_modules|___|__build'

debug = False

fileSplitLevel = {
    '/': 3,
    '/admin-de/reference': 2,
}
pageTemplate = f'{DOC_DIR}/theme/page.cx.html'
webRoot = f'/doc/{VERSION2}'
staticDir = '_static'
extraAssets = [
    f'{DOC_DIR}/theme/theme.css',
    f'{DOC_DIR}/theme/theme.js',
    f'{DOC_DIR}/theme/theme_home.svg',
    f'{DOC_DIR}/theme/theme_info.svg',
    f'{DOC_DIR}/theme/theme_search.svg',
    f'{DOC_DIR}/theme/theme_warning.svg',
    f'{DOC_DIR}/theme/theme_arrow_prev.svg',
    f'{DOC_DIR}/theme/theme_arrow_next.svg',
    f'{DOC_DIR}/theme/theme_arrow_up.svg',
]

includeTemplate = f'{DOC_DIR}/extra_commands.cx.html'

serverPort = 5500
serverHost = '0.0.0.0'

title = 'GBD WebSuite Dokumentation'
subTitle = VERSION3

# wkhtmltopdf options

pdfOptions = {
    'margin-bottom': 20,
    'margin-left': 20,
    'margin-right': 20,
    'margin-top': 20,
    'footer-font-size': 7,
    'footer-right': '[page]',
    'footer-spacing': 5,
}

# apidoc options (see make_api)

apidocWebRoot = f'/apidoc/{VERSION2}'

pydoctorExclude = [
    '___*'
    '_plugins'
    '__pycache__'
    'vendor'
    '*_test.py'
]

pydoctorExtraCss = f'{DOC_DIR}/theme/pydoctor_extra.css'

pydoctorArgs = [
    '--make-html',
    '--project-name', 'GBD WebSuite',
    '--project-version', VERSION2,
    '--docformat', 'google',
    '--theme', 'readthedocs',
    '--html-viewsource-base', 'https://github.com/gbd-consult/gbd-websuite/tree/master/app/gws',
]
