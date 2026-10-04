"""Configuration loading.

This package turns configuration sources (files or Python dicts) into a
validated ``gws.Config`` tree and builds the object tree (``gws.Root``) from it.
It also stores the configured root on disk and loads it back in the server
processes.

Submodules:

- ``gws.config.parser``: reads configuration files (``.cx``, ``.json``,
  ``.yaml``/``.yml``, ``.py``), converts them to plain dicts and validates them
  against the specs. Application configs are handled specially: the server time
  zone is set first, and projects are parsed separately, from the inline
  ``projects`` list, from ``projectPaths`` and from files in ``projectDirs``.
- ``gws.config.loader``: drives the whole process. It creates the specs, runs the
  parser, creates and initializes the root object, collects errors and warnings
  into a ``gws.ConfigResult`` and logs a report. It also stores the root as a
  pickle file, loads it back and keeps the active root in an application global.
- ``gws.config.util``: helpers used by ``configure`` methods of objects to
  create common children (templates, models, finders) and to find providers,
  database providers and source layers.

Errors do not stop the process. The parser and the loader collect them into a
``gws.ConfigContext`` as ``gws.ConfigErrorInfo`` objects, together with the
errors and warnings that objects report while being configured. The result
contains the root object (unless configuration failed or the manifest requires
a strict configuration), the parsed config and all errors and warnings.

If no config path is given, the path is taken from the ``GWS_CONFIG``
environment variable or one of the default paths ``/data/config.cx``,
``/data/config.json``, ``/data/config.yaml``, ``/data/config.py``.

This ``__init__.py`` re-exports the main loader functions and the
``CONFIG_PATH_PATTERN`` used to find config files in directories.

Example::

    import gws.config

    cr = gws.config.configure(config_path='/data/config.cx')
    gws.config.log_report(cr)
    if cr.root:
        gws.config.store(cr.root)

    # later, in a server process
    root = gws.config.load()

Example::

    # parse only, without creating objects
    cr = gws.config.parse(config_path='/data/config.cx')
    for err in cr.errors:
        print(err.message, err.path, err.line)
"""

from .loader import (
    activate,
    configure,
    deactivate,
    initialize,
    parse,
    load,
    get_root,
    store,
    log_report,
)
from .parser import CONFIG_PATH_PATTERN
from . import loader, parser
