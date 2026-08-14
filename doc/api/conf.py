# Configuration file for the Sphinx documentation builder.

import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__) + '/../..')

project = 'GBD WebSuite'
author = 'Geoinformatikbüro Dassau GmbH'
copyright = f'{author} 2006-2026'

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
html_copy_source = False

default_role = 'py:obj'

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
autoapi_keep_files = True

autoapi_ignore = [
    '*___*',
    '*vendor*',
    '*wsgi_app*',
    '*_test*',
    '*_test/*',
    '*_gws*',
    '*geo_info_dok*',
    '*_demo*',
    '*live_config*',
    '*inspire_generate*',
]

autoapi_template_dir = f'{BASE_DIR}/doc/api/_templates/_autoapi'

autoapi_add_toctree_entry = False

autoapi_options = [
    'members',
    # 'inherited-members',
    'undoc-members',
    # 'private-members',
    # 'special-members',
    # 'show-inheritance',
    # 'show-inheritance-diagram',
    # 'show-module-summary',
    # 'imported-members',
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


import hashlib
import io
import pickle

import astroid
import autoapi
import autoapi._mapper


def patch_autoapi():
    """Patch AutoApi's file IO to make incremental builds incremental.

    AutoApi rebuilds its whole world on every run. Two independent mechanisms
    cause that, and both have to be defeated, or a one-line source edit costs a
    full cold build (measured: 34s instead of 4s).

    1. Writing (LazyWriter)

    ``_mapper.output_rst`` opens every generated rst file with mode "wb+" and
    writes it unconditionally, even when the content is identical to the last
    run. That bumps the mtime of all ~430 files, Sphinx compares source mtimes
    against its doctrees, finds everything outdated and re-reads the entire
    corpus. LazyWriter buffers the output in memory and touches the file only if
    the bytes actually differ, so one changed source invalidates one doctree.

    2. Reading (LazyReader)

    ``_mapper.read_file`` parses every source file with astroid on every run in
    which anything changed at all - AutoApi has no per-file cache. astroid is
    expensive: it builds an enriched node model with scopes and name resolution,
    runs its plugin transforms over every node, and follows imports into other
    modules, including the stdlib. That is ~13s for this codebase. LazyReader
    pickles the per-file parse result and returns it on the next run.

    Notes on the implementation:

    - Both patches work by shadowing: ``_mapper`` calls the builtin ``open``
      unqualified, so a module attribute of that name takes precedence.

    - The parse cache is keyed by a hash of the file *content*, not its mtime,
      so a touch is a cache hit and a revert cannot resurrect a stale entry. The
      AutoApi and astroid versions are part of the key, so upgrading either one
      invalidates everything automatically.

    - Pickle, not json: the parsed data contains ``ArgInfo`` namedtuples, which
      json turns back into plain lists, and the templates address them by
      attribute. This is a build cache under our own build directory, so
      unpickling is not a trust issue.

    - The cache lives next to the generated sources (``<temp>/parse``), which
      ties its lifetime to the rest of the build state: doc.py drops that whole
      directory on -no-cache. It holds one entry per content version of every
      file and is never pruned otherwise; a full generation is about 5 MB.
    """

    cache_tag = f'{autoapi.__version__}-{astroid.__version__}'
    orig_read_file = autoapi._mapper.Mapper.read_file

    class LazyWriter(io.BytesIO):
        def __init__(self, path):
            super().__init__()
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            data = self.getvalue()
            try:
                with open(self.path, 'rb') as fp:
                    if fp.read() == data:
                        return super().close()
            except OSError:
                pass
            with open(self.path, 'wb') as fp:
                fp.write(data)
            super().close()

    def lazy_open(path, mode='r', *args, **kwargs):
        if 'b' in mode and ('w' in mode or '+' in mode):
            return LazyWriter(path)
        return open(path, mode, *args, **kwargs)

    def lazy_read_file(self, path, **kwargs):
        try:
            with open(path, 'rb') as fp:
                content = fp.read()
        except OSError:
            return orig_read_file(self, path, **kwargs)

        key = hashlib.sha1(b'\0'.join([path.encode('utf8'), cache_tag.encode('utf8'), content]))
        cache_dir = os.path.join(os.path.dirname(self.app.srcdir), 'parse')
        cache_path = os.path.join(cache_dir, key.hexdigest() + '.pickle')

        try:
            with open(cache_path, 'rb') as fp:
                return pickle.load(fp)
        except (OSError, EOFError, pickle.PickleError):
            pass

        data = orig_read_file(self, path, **kwargs)

        if data is not None:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path + '.tmp', 'wb') as fp:
                pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(cache_path + '.tmp', cache_path)

        return data

    autoapi._mapper.open = lazy_open
    autoapi._mapper.Mapper.read_file = lazy_read_file


patch_autoapi()
