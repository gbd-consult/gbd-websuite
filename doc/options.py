import os

SELF_DIR = os.path.dirname(__file__)

ROOT_DIR = os.path.realpath(SELF_DIR + '/../')
DOC_DIR = SELF_DIR
APP_DIR = ROOT_DIR + '/app'
BUILD_DIR = APP_DIR + '/__build/docbuild'

VERSION, _, _ = open(APP_DIR + '/VERSION').read().strip().rpartition('.')

OPTIONS = {

    # docs options (see lib/vendor/dog)

    'rootDirs': [ROOT_DIR],
    'docPatterns': ['*.doc.md'],
    'assetPatterns': ['*.svg', '*.png'],
    'excludeRegex': 'node_modules|___',

    'logLevel': 'INFO',

    'htmlSplitLevel': 3,
    'htmlPageTemplate': f'{DOC_DIR}/theme/page.cx.html',
    'htmlWebRoot': f'/doc/{VERSION}',
    'htmlStaticDir': '_static',
    'htmlAssets': [
        f'{DOC_DIR}/theme/theme.css',
        f'{DOC_DIR}/theme/theme.js',
    ],

    'outputDir': f'{BUILD_DIR}/doc/{VERSION}',

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
        '--html-output', f'{BUILD_DIR}/apidoc/{VERSION}',
        '--project-name', 'GBD WebSuite',
        '--project-version', VERSION,
        '--no-sidebar',
        '--docformat', 'google',
        '--html-viewsource-base', 'https://github.com/gbd-consult/gbd-websuite/tree/master/app/gws',
    ]
}
