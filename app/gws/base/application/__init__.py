"""Application object.

The application is the single top-level node of the configuration tree, created
from the root configuration. It creates the managers (server, auth, database,
web, actions, search, models, templates, printers, jobs and others), the global
objects (actions, finders, models, templates, printers, exporters, OWS services,
helpers), the default client and finally the projects. Other objects reach these
through ``root.app``.

Submodules:

- ``core``: the application ``Object`` and its root ``Config``.
- ``middleware``: the middleware manager, which orders objects that process each
  web request (for example ``db`` and ``auth``) by their dependencies.
- ``templates``: the built-in templates for the home, error, project and feature
  pages, added to the application templates by default.

Configuration order matters: databases and helpers are configured first, then
storage and auth, then actions, the web site, search, models, templates,
exporters, jobs, printers, OWS services and the client; projects come last,
so they can use all global objects.

Built-in templates for the default subjects and a default printer are added
after the configured ones, so configured templates take precedence. The
template option ``withLogin`` defaults to whether an ``auth`` action is configured.

The middleware manager returns the objects in dependency order, sorted on first
use after a registration. A cyclic or unknown dependency raises ``gws.Error``.

Helpers are configured with ``helpers`` or created on demand by ``app.helper``.
On activation, the application sets the log level and makes the server monitor
watch the configuration files and project directories.

Example::

    {
        title "My application"
        locales [ "de_DE" "en_US" ]
        actions [
            { type web }
            { type map }
        ]
        projectDirs [ "/data/projects" ]
    }

Example::

    project = root.app.project('project_1')
    helper = root.app.helper('upload')
"""

from .core import Config, Object
