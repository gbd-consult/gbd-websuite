import os

SELF_DIR = os.path.dirname(__file__)

ROOT_DIR = os.path.realpath(SELF_DIR + '/../')
DOC_DIR = SELF_DIR

OPTIONS = {
    'rootDirs': [ROOT_DIR],
    'docPatterns': ['*.doc.md'],
    'assetPatterns': ['*.svg', '*.png'],
    'excludeRegex': 'node_modules',

    'docDir': DOC_DIR,

    'htmlSplitLevel': 3,
    'htmlPageTemplate': DOC_DIR + '/theme/page.cx.html',
    'htmlWebRoot': '/docs/8.0.0',
    'htmlStaticDir': '_static',
    'htmlAssets': [
        DOC_DIR + '/theme/theme.css',
        DOC_DIR + '/theme/theme.js',
    ],

    'outputDir': ROOT_DIR + '/app/__build/docs/8.0.0',

    'serverPort': 5500,
    'serverHost': '0.0.0.0',

    'title': 'GBD WebSuite',
    'subTitle': '8.0.0',
}
