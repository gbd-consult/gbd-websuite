import os
import sys
import re

sys.path.insert(0, os.path.dirname(__file__))
import util

project = 'GBD WebSuite'
copyright = '2017-2022, Geoinformatikbüro Dassau GmbH'
author = 'Geoinformatikbüro Dassau GmbH'
version = util.VERSION
release = util.VERSION

DOC_ROOT = util.DOC_ROOT
GEN_ROOT = util.GEN_ROOT

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    # 'sphinx.ext.autosummary',
    # 'sphinx.ext.doctest',
    # 'sphinx.ext.intersphinx',
    # 'sphinx.ext.todo',
    # 'sphinx.ext.coverage',
    # 'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
]

autoclass_content = "class"

autodoc_mock_imports = [
    'fiona',
    'ldap',
    'mapproxy',
    'osgeo',
    'PIL',
    'psutil',
    'psycopg2',
    'pycountry',
    'PyPDF2',
    'pyproj',
    'shapely',
    'svgis',
    'uwsgi',
    'wand',
    'werkzeug',
]

autodoc_member_order = 'bysource'

# autosummary_generate = True

napoleon_numpy_docstring = False
napoleon_use_rtype = False

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'

html_theme = "sphinx_rtd_theme"
html_theme_options = {}
html_static_path = ['_static']
html_sidebars = {}
html_show_sourcelink = False

keep_warnings = True


##

def format_special(app, docname, source):
    m = re.match('^books/(.+?)/([a-z]+)', docname)
    book = m.group(1) if m else ''
    lang = m.group(2) if m else ''
    source[0] = util.format_special(source[0], book, lang)


def setup(app):
    app.add_stylesheet('extras.css')
    app.add_javascript('extras.js')
    app.connect('source-read', format_special)


def pre_build():
    util.clear_output()
    util.cleanup_rst()

    util.make_config_ref('en')
    util.make_config_ref('de')

    util.make_cli_ref('en')
    util.make_cli_ref('de')
    #
    # util.make_autodoc()


def post_build():
    util.make_help('en')
    util.make_help('de')


##

if __name__ == '__main__':
    fn = sys.argv[1]
    if fn == 'pre':
        pre_build()
    if fn == 'post':
        post_build()
