"""Projects.

A project is a self-contained map application inside the GWS application. It has
its own main map and optional overview map, and its own actions, finders, models,
printers, exporters, templates, OWS services, client settings and assets. Projects
are configured in the ``projects`` list of the application config; the client
addresses a project by its uid.

Submodules:

- ``core``: the project object (``gws.Project``) with its ``Config`` and ``Props``.
- ``action``: the ``project`` action, whose ``projectInfo`` command returns the
  project props, the locale and the current user to the client.

Most project settings complement the application settings. Printers and exporters
are offered along with the global ones; the client gets the printers the user can
use, project printers first, or else the application default printer. The project
assets directory is checked before the global one, and the project metadata is
merged with the application metadata. A project ``client`` replaces the
application client. If no locales are configured, the application locales are
used. The project title is taken from the ``title`` option, then from the
metadata, then from the uid.

The project description sent to the client is rendered from a template with the
subject ``project.description``. The project home page (``/project/<uid>``) is
rendered from a template with the subject ``project.home`` by the ``web`` action.

Example::

    {
        actions [
            { type "project" }
        ]
        projects [
            {
                uid "project_1"
                title "Project 1"
                map {
                    layers [ ... ]
                }
                templates [
                    { subject "project.description" type "html" path "/data/description.cx.html" }
                ]
            }
        ]
    }
"""

from .core import Config, Object, Props

