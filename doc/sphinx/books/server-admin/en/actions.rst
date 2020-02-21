Server Actions
==============

GBD WebSuite supports lots of different commands. In your application and project configuration you may decide which you want to use in your particular installation. Each action configuration requires at least the ``type`` property (the first word of the command name, e.g. for the command ``mapRenderXyz``, the type will be ``map``), and, optionally, an ``access`` block (see :doc:`auth`), telling which roles have permission to *execute* this action type. Some actions require additional options.

Here's a quick overview of supported actions (see :doc:`configref` for configruation details).

TABLE
*alkissearch* ~  provides interface to the German Cadastre Data (`ALKIS <http://www.adv-online.de/Products/Real-Estate-Cadastre/ALKIS/>`_). In particular, there are commands to search for Cadastre Parcels (*Flurstücke*) by their address, location, owner name etc.
*alkisgeocoder* ~  geocoding for ALKIS data
*asset* ~ handles dynamic assets (see :doc:`web`)
*auth* ~ handles authorization requests, like login or logout (see :doc:`auth`)
*dimension* ~ supports the Dimension tool in the client
*edit* ~ provides support for editing operations (like "update feature" or "delete feature")
*map* ~ generates map imagery for projects and layers in different formats
*ows* ~ provides OWS services like WMS and WFS
*printer* ~ handles printing
*project* ~ returns project description and configuration data
*search* ~ handles searching (see :doc:`search`)
*storage* ~ handles storage requests (see :doc:`storage`)
/TABLE


Helpers
-------

Helpers are specific modules in the WebSuite shared by certain actions. Helpers must be enabled and configured separately in the main application config. Currently the following helpers are available

TABLE
*alkis* ~  used by ``alkis*`` actions
*csv* ~  used by ``alkis*`` actions
*storage* ~  used by the ``storage`` action
/TABLE
