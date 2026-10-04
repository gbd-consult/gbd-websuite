"""Printing.

Prints maps and templates to PDF or PNG. Printing runs as a background job:
the client sends a print request, polls the job status and downloads the
result when the job is complete.

Submodules:

- ``core``: the default printer (``gws.Printer``), a print template offered to
  users, with a title, quality levels (DPI) and optional models whose fields
  appear as input fields in the print dialog. Printers are configured in the
  application and in projects. If no quality levels are configured, a single
  ``default`` level is used. The title defaults to the template title. The
  client props include the model the user can write, if any.
- ``manager``: the printer manager (``gws.PrinterManager``), available as
  ``root.app.printerMgr``. Starts print jobs and runs prints synchronously.
  The request is passed to the job as a pickle file.
- ``worker``: the job worker that executes a print request. It resolves the
  project and the printer, determines the output format and DPI (limited by
  the printer quality levels), converts the map planes sent by the client
  (raster and vector layers, bitmaps, data URLs, features, SVG soup) into render
  planes, renders the template and stores the result in the job.
- ``action``: the ``printer`` action with the API commands ``printerStart``,
  ``printerStatus``, ``printerCancel`` and ``printerOutput``, and the
  ``printerPrint`` CLI command.

A print request (``gws.PrintRequest``) is either of type ``template``, which uses
a configured printer, or ``map``, which renders the map with a temporary map
template of the requested output size.

Example::

    actions+ { type "printer" }

    printers+ {
        title "A4 portrait"
        template { type "html" path "a4.cx.html" mapSize [ "150mm" "100mm" ] }
        qualityLevels [
            { dpi 90 name "screen" }
            { dpi 300 name "print" }
        ]
    }
"""

from .core import Config, Object, Props
from . import manager
