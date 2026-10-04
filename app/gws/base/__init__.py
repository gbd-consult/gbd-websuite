"""Base classes of the configurable objects.

This package holds the base classes, managers and actions of the objects that
make up a GBD WebSuite application. Most interfaces they implement are declared
in the ``types.pyinc`` files of the subpackages and exported as ``gws.*``
types. Concrete object types (layer types, auth providers, templates, model
fields and so on) live in ``gws.plugin`` and subclass the classes here.

Subpackages:

- ``action``: server actions, the base class of all actions and the action manager.
- ``application``: the application object, the root node of the object tree.
- ``auth``: authentication and authorization: the auth manager, base classes
  for auth methods, providers, MFA adapters and session managers, and users.
- ``client``: browser client configuration (UI elements, options) and client bundles.
- ``database``: database providers and the database layer, model and auth provider base classes.
- ``edit``: feature editing, the ``edit`` action and helper.
- ``exporter``: feature export to files, run as background jobs.
- ``feature``: the feature implementation.
- ``grabber``: raster grabbers, which fetch, warp, cache and serve layer images.
- ``job``: background jobs and the job manager.
- ``layer``: base classes and utilities for map layers.
- ``legend``: base classes for layer legends.
- ``map``: maps and the ``map`` action.
- ``metadata``: OWS and ISO 19115 metadata.
- ``model``: data models and their fields, values, validators and widgets.
- ``ows``: common base for OWS clients and servers.
- ``printer``: printing to PDF or PNG.
- ``project``: projects and the ``project`` action.
- ``search``: search queries, finders and the search manager.
- ``shape``: shapes, geometries with a CRS.
- ``storage``: storage for data saved by users.
- ``template``: templates and the template manager.
- ``web``: the WSGI application, web site settings and the ``web`` action.

How they relate:

The application (``application``) is created from the root configuration and
creates the managers (auth, database, models, search, templates, printers,
jobs, web and others), the global objects and the projects. Other objects
reach them through ``root.app``. A project (``project``) has a map (``map``)
with a tree of layers (``layer``). Layers have legends (``legend``),
metadata (``metadata``), templates (``template``), finders (``search``) and
models (``model``). Models read and write features (``feature``) with
geometries (``shape``) from their sources, for example databases
(``database``). Image layers render through grabbers (``grabber``). Actions
(``action``) expose commands to the client; long running tasks such as
printing (``printer``) and exporting (``exporter``) run as background jobs
(``job``). Each web request passes through the WSGI application (``web``),
which runs the middleware (for example ``auth``) and dispatches the command
to an action.

``gws.base`` imports from ``gws.core``, ``gws.lib``, ``gws.gis``,
``gws.config``, ``gws.spec`` and ``gws.server``, never from ``gws.plugin``.

Example::

    import gws
    import gws.base.layer
    import gws.base.shape
    import gws.lib.crs

    shape = gws.base.shape.from_xy(100, 200, gws.lib.crs.WEBMERCATOR)

    @gws.ext.object.layer('my')
    class Object(gws.base.layer.image.Object):
        def configure(self):
            self.configure_layer()
"""
