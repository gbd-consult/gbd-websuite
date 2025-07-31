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
    '/user-de': 2,
    '/user-de/sidebar.editieren': 3,
    '/admin-de': 2,
    '/admin-de/themen': 3,
}
pageTemplate = f'{DOC_DIR}/theme/page.cx.html'
webRoot = f'/doc/{VERSION2}'
staticDir = '_static'
extraAssets = [
    f'{DOC_DIR}/theme/theme.css',
    f'{DOC_DIR}/theme/theme.js',
    f'{ROOT_DIR}/data/web/gws_logo.svg',
]

brandLogo = "gws_logo.svg"
brandURL = "https://gbd-consult.de"
brandText = "&copy; Geoinformatikbüro Dassau GmbH 2006-2025"

includeTemplate = f'{DOC_DIR}/extra_commands.cx.html'

serverPort = 5500
serverHost = '0.0.0.0'

title = 'GBD WebSuite'
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

del os
