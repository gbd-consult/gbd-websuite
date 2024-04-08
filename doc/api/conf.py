# Configuration file for the Sphinx documentation builder.

import os
import sys

BASE_DIR = os.getenv('BASE_DIR')

project = 'GBD WebSuite'
author = 'Geoinformatikbüro Dassau GmbH'
copyright = f'{author} 2006–2024'

with open(f'{BASE_DIR}/app/VERSION') as fp:
    release = fp.read().strip()

version = '.'.join(release.split('.')[:-1])

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_static_path = ['_static']
html_title = f"{project} {release}"
html_logo = f"{BASE_DIR}/data/web/gws_logo.svg"
html_css_files = ['custom.css']

extensions = [
    'sphinx.ext.napoleon',
    'autoapi.extension',
]

# AutoApi configuration
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html

autoapi_dirs = [
    f'{BASE_DIR}/app/gws',
]

autoapi_root = f'py'
autoapi_keep_files = False

autoapi_ignore = [
    '*___*',
    '*vendor*',
    '*wsgi_app*',
    '*_test*',
    '*_test/*',
]

autoapi_template_dir = '_templates/_autoapi'

autoapi_add_toctree_entry = False

autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'imported-members',
]

"""
Support our custom :source: role

This role provides module-level github links. 
It is used in the apidoc "module.rst" template like this:

    **Source code:** :source:`{{ obj.name }}<{{ obj.obj.relative_path }}>`

Inspired by https://github.com/python/cpython/blob/main/Doc/tools/extensions/pyspecific.py
"""

GWS_GITHUB = 'https://github.com/gbd-consult/gbd-websuite/tree/master/app'

import sphinx.util.nodes
import docutils.utils
import docutils.nodes


def setup(app):
    app.add_role('source', source_role)
    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }


def source_role(typ, rawtext, text, lineno, inliner, **kwargs):
    has_t, title, target = sphinx.util.nodes.split_explicit_title(text)
    title = docutils.utils.unescape(title)
    target = docutils.utils.unescape(target)
    ref = docutils.nodes.reference(rawtext, title, refuri=f'{GWS_GITHUB}/{target}')
    return [ref], []
