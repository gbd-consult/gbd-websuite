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

html_theme = 'sphinxdoc'
html_static_path = ['_static']
html_title = f"{project} {release}"
html_logo = f"{BASE_DIR}/data/web/gws_logo.svg"
html_css_files = ['custom.css']

# https://www.sphinx-doc.org/en/master/usage/theming.html#builtin-themes

html_theme_options = {
    # 'nosidebar': False,
    # 'sidebarwidth': False,
    # 'body_min_width': False,
    # 'body_max_width': False,
    # 'navigation_with_keys': False,
    # 'enable_search_shortcuts': False,
    # 'globaltoc_collapse': False,
    # 'globaltoc_includehidden': False,
    # 'globaltoc_maxdepth': 1,
}


extensions = [
    'sphinx.ext.napoleon',
    'autoapi.extension',
]

# Napoleon configuration
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True


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
