import os

SELF_DIR = os.path.dirname(__file__)

ROOT_DIR = os.path.realpath(SELF_DIR + '/../')
DOC_DIR = SELF_DIR
APP_DIR = ROOT_DIR + '/app'
BUILD_DIR = APP_DIR + '/__build'
DOC_BUILD_DIR = BUILD_DIR + '/docbuild'

VERSION, _, _ = open(APP_DIR + '/VERSION').read().strip().rpartition('.')

OPTIONS = {

    # docs options (see lib/vendor/dog)

    'rootDirs': [ROOT_DIR],
    'docPatterns': ['*.doc.md'],
    'assetPatterns': ['*.svg', '*.png'],
    'excludeRegex': 'node_modules|___|__build',

    'debug': False,

    'htmlSplitLevel': 3,
    'pageTemplate': f'{DOC_DIR}/theme/page.cx.html',
    'webRoot': f'/doc/{VERSION}',
    'staticDir': '_static',
    'extraAssets': [
        f'{DOC_DIR}/theme/theme.css',
        f'{DOC_DIR}/theme/theme.js',
        f'{DOC_DIR}/theme/theme_home.svg',
        f'{DOC_DIR}/theme/theme_info.svg',
        f'{DOC_DIR}/theme/theme_warning.svg',
    ],

    'includeTemplate': f'{DOC_DIR}/extra_commands.cx.html',

    'outputDir': f'{DOC_BUILD_DIR}/doc/{VERSION}',

    'serverPort': 5500,
    'serverHost': '0.0.0.0',

    'title': 'GBD WebSuite',
    'subTitle': VERSION,

    # apidoc options (see make_api)

    'docDir': DOC_DIR,
    'appDir': APP_DIR,
    'buildDir': BUILD_DIR,

    'apidocWebRoot': f'/apidoc/{VERSION}',

    'pydoctorExclude': [
        '___*',
        '_plugins',
        '__pycache__',
        'vendor',
        '*_test.py',
    ],

    'pydoctorArgs': [
        '--make-html',
        '--html-output', f'{DOC_BUILD_DIR}/apidoc/{VERSION}',
        '--project-name', 'GBD WebSuite',
        '--project-version', VERSION,
        '--no-sidebar',
        '--docformat', 'google',
        '--html-viewsource-base', 'https://github.com/gbd-consult/gbd-websuite/tree/master/app/gws',
    ]
}
