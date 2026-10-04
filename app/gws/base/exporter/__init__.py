"""Feature export.

Exports features selected in the client to files, such as Shapefile, GeoJSON,
GML, KML or CSV. Exports run as background jobs. Concrete exporters, one per
format, live in ``gws.plugin.exporter``.

Submodules
----------

- ``core`` - the base exporter object with its configuration and props.
  Subclasses declare which kinds of data they support and implement ``run``.
  ``withMultiLayer`` is only enabled if the subclass supports multiple layers.
- ``manager`` - the exporter manager (``root.app.exporterMgr``). It lists the
  exporters available to a user, starts export jobs and runs exports directly
  from the command line.
- ``worker`` - the job worker. It loads the requested features, finds the
  exporter and calls its ``run`` method, reporting progress to the job.
- ``action`` - the ``exporter`` action with the API commands to start, poll,
  cancel and download an export, and the ``gws exporter export`` command line
  command.
- ``util`` - helpers for exporters: grouping features by model with consistent
  columns, geometry type and CRS, and a complete export with a GDAL vector driver.

Exporters are configured on projects and on the application (``exporters``).
For a project, the exporters of the project and of the application that the
user can use are offered; of several exporters with the same title, only the
first one is used.

Export flow
-----------

1. The client sends ``exporterStart`` with a ``gws.ExportRequest`` (exporter
   uid and feature props). The manager creates a job for the worker and
   schedules it. The request is passed to the worker through a pickled file
   in the ephemeral directory.
2. The worker loads the features. Depending on the ``exportStrategy`` of
   their model, features are read from the model by uid or built from the
   props sent by the client.
3. The exporter writes the files and fills the ``gws.ExportResult``: the path
   of the result file, its mime type and counts. The GDAL helper in ``util``
   zips the files if there is more than one.
4. The client polls ``exporterStatus``. When the job is complete, the status
   contains the counts and a URL for ``exporterOutput``, which returns the file.

In the ``util`` helpers, a feature that does not fit its group (no geometry, a different geometry type
or a different CRS) is skipped with an error in the result, unless the
exporter allows it with ``withNoGeometry``, ``withMixedGeometry`` or
``withMixedCrs``.

Example::

    actions+ {
        type "exporter"
    }

    exporters+ {
        type "geojson"
        title "GeoJSON"
        target "download"
        access "allow all"
    }

    exporters+ {
        type "gml"
        title "GML (one file)"
        target "download"
        withMultiLayer true
        access "allow all"
    }

From the command line, with a request stored as JSON::

    gws exporter export --request request.json --output export.zip

Implementing an exporter::

    @gws.ext.object.exporter('myformat')
    class Object(gws.base.exporter.Object):
        supportsVector = True
        supportsRaster = False
        supportsMultiLayer = False

        def run(self, ea, er):
            gws.base.exporter.util.run_gdal_vector_export('MyDriver', 'application/x-my', ea, er)
"""

from .core import Config, Object, Props
from . import manager, util

