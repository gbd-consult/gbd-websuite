"""Built-in plugins.

Plugins provide the concrete object types of GBD WebSuite: layer types, auth
methods and providers, model fields, templates, OWS services and so on. Each
plugin is a package that subclasses the base classes from ``gws.base`` and
registers its classes with the ``gws.ext`` decorators, for example
``@gws.ext.object.layer('tile')`` together with
``@gws.ext.config.layer('tile')``. The registered name is the ``type`` used in
the configuration.

Plugins are not imported by other code. The spec generator scans every
directory under ``gws/plugin`` (and the plugin directories listed in the
application manifest) and records the extension classes in the specs. When
the configuration refers to an extension type, the class is looked up in the
specs and its module is imported on demand. A plugin can also contain client
code in a ``js`` directory, which is built into the client bundle; some
plugins (``identify_tool``, ``location_tool``) are client-only and have no
Python code.

Plugins import from ``gws.base`` and the lower layers, and from other plugins
where one builds on another (for example ``gekos`` uses ``alkis``, and
``alkis`` and ``qgis`` use ``postgres``).

Subpackages:

Authentication

- ``account``: user accounts stored in a database table, with administration and onboarding.
- ``auth_method``: authentication methods (``web``, ``basic``, ``token``).
- ``auth_mfa``: multi-factor authentication adapters.
- ``auth_provider``: authentication providers.
- ``auth_session_manager``: session managers.

Data sources and layers

- ``geojson``: GeoJSON support.
- ``mbtiles_layer``: MBTiles layer.
- ``ows_client``: clients for remote OGC web services (WMS, WMTS, WFS).
- ``postgres``: PostgreSQL/PostGIS provider, layers, models and finders.
- ``qgis``: QGIS projects as sources for layers, search, models, legends and print templates.
- ``raster_layer``: raster layers from georeferenced image files.
- ``tile_layer``: tile layers from XYZ tile services.

Models

- ``model_field``: model field types.
- ``model_validator``: model validators.
- ``model_value``: model values.
- ``model_widget``: model widgets.

Output and services

- ``exporter``: feature exporters to vector file formats.
- ``legend``: legend types.
- ``ows_server``: OGC web services (WMS, WMTS, WFS, CSW) provided by GBD WebSuite.
- ``template``: template types.

Search

- ``gbd_geoservices``: GBD Geoservices search.
- ``nominatim``: place and address search with Nominatim.

Client tools

- ``annotate_tool``: annotate tool.
- ``dimension``: dimension tool.
- ``identify_tool``: identify tool (client only).
- ``location_tool``: shows the user's location from the browser geolocation (client only).
- ``mapshare``: map share tool.
- ``select_tool``: feature selection tool.

Helpers and storage

- ``csv_helper``: CSV helper.
- ``email_helper``: email helper.
- ``storage_provider``: storage providers for data saved by users.
- ``upload_helper``: helper for chunked file uploads.
- ``xml_helper``: helper for custom XML namespaces.

Administration

- ``admin_action``: administration pages (object inspector, cache viewer).

Applications

- ``alkis``: ALKIS cadastre search.
- ``gekos``: GekoS-Bau integration.
- ``qfieldcloud``: QField Cloud plugin.

Example::

    import gws
    import gws.base.layer

    @gws.ext.config.layer('my')
    class Config(gws.base.layer.Config):
        pass

    @gws.ext.object.layer('my')
    class Object(gws.base.layer.image.Object):
        def configure(self):
            self.configure_layer()

Configuration example, using plugin types::

    auth.providers+ { type "file" path "/data/passwd.json" }

    map.layers+ {
        type "tile"
        provider.url "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    }
"""
