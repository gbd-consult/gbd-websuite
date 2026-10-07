"""QField Cloud plugin.

This plugin emulates the QFieldCloud API, so that the QField mobile app can
download projects from GWS and synchronize edits back. The QGIS projects are
packaged according to the settings made with the QFieldSync QGIS plugin.

.. rubric:: Submodules

- ``action``: the ``qfieldcloud`` action. It receives API requests and passes each one to a new request handler.
- ``action_base``: the base class of the action. It holds the QField projects, caches their capabilities, creates packages (also in background jobs) and manages the package and cache directories.
- ``action_handler``: the request handler. It dispatches an API request to a route method, authenticates the client by token and stores incoming deltas. It holds the state of a single request only and accesses persistent data via the action.
- ``api``: data classes and enums mirroring the QFieldCloud API objects (``swagger.yaml``), plus a few objects used by the client that are not in the specification.
- ``auth``: the ``qfieldcloud`` authorization method, created and registered by the action. It has the fixed uid ``gws.plugin.qfieldcloud.auth``.
- ``caps``: reads the QFieldSync project and layer properties from the QGIS project and decides, per layer, whether it is packaged for editing, packaged as a base map or removed.
- ``cli``: the ``qfieldcloudPackage`` command, which creates a package into a directory.
- ``core``: the configuration and object of a single QField project.
- ``packager``: writes a package (GeoPackage data, base maps, media files, modified QGIS project).
- ``patcher``: applies changes ("deltas") and file uploads from QField to the database.

.. rubric:: Design

The action holds a list of QField projects (``core.QfcProject``), each built on
a QGIS project. For each QField project, ``caps.Parser`` creates a ``caps.Caps``
object with the layer and model map; the action caches it until the QGIS project
source changes. Packaging and patching both work on these capabilities.

The action answers requests under the raw command ``qfieldcloudApi``. A rewrite
rule maps a regular address to that command and names the GWS project::

    web.sites+ {
        rewriteRules+ {
            pattern "^/qfc/(.*)"
            target "/_/qfieldcloudApi/projectUid/my_project/$1"
        }
    }

QField users enter the resulting address (``https://example.com/qfc``) as the
server and log in with their GWS credentials. The action brings its own
authorization method; an auth provider and a session manager must be configured.
Requests other than the public status and login routes require an
``Authorization: Token ...`` header with the session token.

.. rubric:: Configuration

The plugin exposes an action of type ``qfieldcloud``. In the action config, you can define multiple ``projects``, each representing a QField project::

    actions+ {
        type "qfieldcloud"

        projects+ {
            title "My Project"
            provider.path "/path/to/file.qgs"
        }
    }

.. rubric:: Downloading data

Data flow GWS -> QField, also called "packaging".

When QField requests a package job, the action runs the packager in a background job.
In the given QGIS project, the plugin looks for Postgres layers marked as "offline editable", fetches their data and writes each table into a GeoPackage file. A modified QGIS project file is also created, pointing to the GeoPackage layers instead of the original Postgres layers. Layers marked "remove" are removed from the project, and empty layer groups are dropped.

For each offline table, a Model can be configured to customize field selection and data filtering. Models are defined in the project configuration and matched by table name. By default, the plugin uses a generic Postgres model that includes all fields and all features.

Base map layers (a map theme or a single layer, as selected in QFieldSync) are rendered through QGIS into a single raster image covering the area of interest at the configured maximum zoom level. The rendered images are cached in the project cache directory for ``mapCacheLifeTime``.

Directories listed in the QFieldSync attachment, data and copy settings are added to the package as media files.

.. rubric:: Uploading data

Data flow QField -> GWS, also called "patching".

For each incoming "delta" payload from QField, the plugin extracts the created, updated and deleted features and passes them to the respective Model. The Model is responsible for applying the changes to the Postgres database. The payload is stored for an hour, so that QField can poll its status.

.. rubric:: File uploads

If a Model supports file uploads, it should contain a virtual file field with ``nameColumn`` and ``contentColumn``::

    actions+ {
        type "qfieldcloud"

        projects+ {
            title "My Project"
            provider.path "/path/to/file.qgs"

            models+ {
                type "postgres"
                ...
                fields+ {
                    type "file"
                    name "virtual_file_field"
                    contentColumn "file_content"
                    nameColumn "file_name"
                }
            }
        }
    }

QField sends uploads in two steps: first, the file path is included along with the feature changes in the delta payload. Later, the actual file content is uploaded in a separate request. The plugin matches the file content to the respective feature based on the ``nameColumn`` value and writes it into ``contentColumn``.

.. rubric:: Extending

Override the packager, patcher and handler classes to customize packaging, patching and request handling.
In your custom action class, override ``get_packager()``, ``get_patcher()`` and ``get_handler()`` methods to return your custom classes.
Route methods of a custom handler are marked with the ``action_handler.route`` decorator.

Example::

    import gws.plugin.qfieldcloud.action
    import gws.plugin.qfieldcloud.action_handler
    import gws.plugin.qfieldcloud.patcher

    class MyPatcher(gws.plugin.qfieldcloud.patcher.Object):
        def commit_operations_for_model(self, me, ops):
            ...
            super().commit_operations_for_model(me, ops)

    class MyHandler(gws.plugin.qfieldcloud.action_handler.Handler):
        @gws.plugin.qfieldcloud.action_handler.route('GET api/v1/status')
        def on_get_status(self):
            ...
            return super().on_get_status()

    class MyAction(gws.plugin.qfieldcloud.action.Object):
        def get_patcher(self):
            return MyPatcher()

        def get_handler(self):
            return MyHandler(self)
"""

from . import (
    action,
    action_base,
    action_handler,
    packager,
    patcher,
    caps,
)

__all__ = [
    'action',
    'action_base',
    'action_handler',
    'packager',
    'patcher',
    'caps',
]
