# Components :/dev-en/overview/components

The server code is the package `gws` in `app/gws`. Each package links to its API documentation, built with `make.sh doc-api`.

## Framework ::

| Package | Contents |
|---------|----------|
| <% pyapi('gws') %> | Basic types and the interfaces of all components, generated from the `.pyinc` files of the packages. Every module imports it with `import gws`. |
| <% pyapi('gws.config') %> | Reads configuration files in all supported formats, validates them against the specs, builds the object tree, stores it and loads it in the worker processes. <% pyapi('gws.config.util', 'gws.config.util') %> has helpers that configure common children, like templates, models and finders. |
| <% pyapi('gws.core') %> | The object tree implementation behind <% pyapi('gws.Node', 'gws.Node') %> and <% pyapi('gws.Root', 'gws.Root') %>. Also general utilities (<% pyapi('gws.core.util', 'gws.u') %>), logging (<% pyapi('gws.core.log', 'gws.log') %>), debugging helpers (<% pyapi('gws.core.debug', 'gws.debug') %>), constants such as system paths and role names (<% pyapi('gws.core.const', 'gws.c') %>), and environment variables (<% pyapi('gws.core.env', 'gws.env') %>). |
| <% pyapi('gws.ext') %> | The decorators that register extension types and commands. They do nothing at runtime, the spec generator reads them from the source. |
| <% pyapi('gws.server') %> | Configuration and control of the embedded servers (nginx, uWSGI, rsyslog), the `gws server` commands, the monitor that reloads the server on configuration changes, and the spooler for background jobs. |
| <% pyapi('gws.spec') %> | The spec generator, which parses the source code, and the spec runtime, which validates configurations and requests, converts special types and imports extension classes on demand. |
| <% pyapi('gws.test') %> | The test runner behind `make.sh test`, the in-container runner, the mock HTTP server and the utilities used in tests. |

## Base ::

<% pyapi('gws.base', 'gws.base') %> contains the core components. They are always loaded, and most of them define a base class that plugins extend.

| Package | Contents |
|---------|----------|
| <% pyapi('gws.base.action') %> | The base action class and the action manager, which finds the action for a command, validates the request and checks permissions. Also the `gws action` CLI commands. |
| <% pyapi('gws.base.application') %> | The application object, the first node of the tree. It configures all global components in a fixed order and holds the managers, like <% pyapi('gws.Application.actionMgr') %>, <% pyapi('gws.Application.authMgr') %> or <% pyapi('gws.Application.databaseMgr') %>. |
| <% pyapi('gws.base.auth') %> | Users and roles, the authorization manager, base classes for authentication methods (how credentials arrive), providers (where users come from), multi-factor adapters and sessions. |
| <% pyapi('gws.base.client') %> | The configuration of the client UI: which elements (toolbar buttons, sidebar tabs, and so on) a project shows, and their options. |
| <% pyapi('gws.base.database') %> | Database providers and connections (SQLAlchemy), and base classes for database-backed layers, models and user sources. |
| <% pyapi('gws.base.edit') %> | The `edit` action, which lets the client list, create, update and delete features of editable models. |
| <% pyapi('gws.base.exporter') %> | Exporters that write features to files, the export worker and the `exporter` action. |
| <% pyapi('gws.base.feature') %> | The feature object: attributes, geometry, the raw source record and the views rendered by templates. |
| <% pyapi('gws.base.grabber') %> | Raster images for layers: fetches them from sources, aligns them to tile grids, reprojects them and caches tiles. |
| <% pyapi('gws.base.job') %> | The job manager: creates jobs, stores their state in sqlite, runs them in the spooler and reports their status and results. |
| <% pyapi('gws.base.layer') %> | The base layer class, layer groups, base classes for raster and vector layers, and the configuration of layer trees from source layers. |
| <% pyapi('gws.base.legend') %> | The base legend class, which renders a legend for a layer, and functions to combine several legends into one image. |
| <% pyapi('gws.base.map') %> | The map object, which defines the coordinate system, extent and resolutions of a project, and the `map` action, which renders images, tiles, legends and features of layers. |
| <% pyapi('gws.base.metadata') %> | Metadata of projects, layers and services, based on ISO 19115 and the OGC service standards, with INSPIRE support. |
| <% pyapi('gws.base.model') %> | Data models and fields: reading features from a source, converting them for the client, and validating and writing them back. The module docstring describes the data flow. |
| <% pyapi('gws.base.ows') %> | OGC web services. <% pyapi('gws.base.ows.client', 'ows.client') %> parses capabilities and queries WMS, WFS and WMTS sources. <% pyapi('gws.base.ows.server', 'ows.server') %> implements WMS, WMTS, WFS and CSW services for configured projects. |
| <% pyapi('gws.base.printer') %> | Printers and the print worker, which renders a map and a template into a PDF or image in a background job. |
| <% pyapi('gws.base.project') %> | Projects: a map, a client configuration and project-specific actions, finders, models, templates and printers. Also the `projectInfo` command, which sends a project to the client. |
| <% pyapi('gws.base.search') %> | The search manager, which runs a search query over all finders a user can access, and the base finder class. Finders search layers, models or external services. |
| <% pyapi('gws.lib.shape') %> | Geometries in a coordinate system, based on Shapely, with conversions from and to WKT, WKB, GeoJSON and extents. |
| <% pyapi('gws.base.storage') %> | A key-value store where the client saves user data, like selections or annotations, by category. Storage providers implement the backend. |
| <% pyapi('gws.base.template') %> | The base template class and the template manager, which finds a template by subject (for example `feature.label`) along the tree. |
| <% pyapi('gws.base.web') %> | Web sites with their rewrite rules, CORS and SSL settings, request parsing (<% pyapi('gws.WebRequester', 'gws.WebRequester') %>), the WSGI application and the `web` action for assets, pages and downloads. |

## GIS ::

| Package | Contents |
|---------|----------|
| <% pyapi('gws.gis.cache') %> | Tile cache management: status, seeding, cleanup, and the `gws cache` commands. |
| <% pyapi('gws.gis.render') %> | Renders a map view from layers into raster and SVG planes, and combines them into an image or HTML. Used by map and print templates and by OWS services. |
| <% pyapi('gws.gis.source') %> | Utilities for source layers, the layers of an external source like a WMS service or a QGIS project: filtering and combining their extents and coordinate systems. |
| <% pyapi('gws.gis.zoom') %> | Computes resolutions and scales for maps and layers from the zoom configuration. |

## Libraries ::

<% pyapi('gws.lib', 'gws.lib') %> contains libraries that do not depend on the object tree.

| Package | Contents |
|---------|----------|
| <% pyapi('gws.lib.bounds') %> | Extents with a coordinate system: transforming, combining, buffering. |
| <% pyapi('gws.lib.cli') %> | Helpers for command line scripts. |
| <% pyapi('gws.lib.cql') %> | CQL2 filter expressions. |
| <% pyapi('gws.lib.crs') %> | Coordinate systems: lookup by code or name, axis order, transformations. |
| <% pyapi('gws.lib.datetimex') %> | Dates, times, time zones and durations. |
| <% pyapi('gws.lib.dynimport') %> | Importing modules from file paths, used for extension classes and plugins. |
| <% pyapi('gws.lib.extent') %> | Plain extents: parsing, intersecting, computing centers and sizes. |
| <% pyapi('gws.lib.font') %> | Font installation for rendering. |
| <% pyapi('gws.lib.gdalx') %> | A wrapper around GDAL and OGR for raster and vector files. |
| <% pyapi('gws.lib.gml') %> | GML parsing and writing. |
| <% pyapi('gws.lib.grid') %> | Tile grids: resolutions per level, tile ranges for an extent. |
| <% pyapi('gws.lib.htmlx') %> | HTML rendering to PDF or PNG with wkhtmltopdf. |
| <% pyapi('gws.lib.image') %> | Images, a wrapper around Pillow. |
| <% pyapi('gws.lib.inifile') %> | Reading and writing ini files. |
| <% pyapi('gws.lib.intl') %> | Locales and formatting of dates, times and numbers. |
| <% pyapi('gws.lib.jsonx') %> | Reading and writing JSON. |
| <% pyapi('gws.lib.mapserver') %> | Raster rendering with MapServer. |
| <% pyapi('gws.lib.mime') %> | MIME types. |
| <% pyapi('gws.lib.net') %> | HTTP requests and URL handling. |
| <% pyapi('gws.lib.osx') %> | Running commands and working with files and processes. |
| <% pyapi('gws.lib.otp') %> | One-time passwords (HOTP, TOTP) for multi-factor authentication. |
| <% pyapi('gws.lib.password') %> | Password hashing, checking and generation. |
| <% pyapi('gws.lib.pdf') %> | PDF utilities: merging, overlaying, converting pages to images. |
| <% pyapi('gws.lib.sa') %> | SQLAlchemy imports in one place. |
| <% pyapi('gws.lib.sqlitex') %> | A wrapper around the SQLite driver. |
| <% pyapi('gws.lib.style') %> | Feature styles: parsing CSS-like style values and icons. |
| <% pyapi('gws.lib.svg') %> | Building SVG fragments from shapes and styles. |
| <% pyapi('gws.lib.text') %> | Text utilities: base64, data URLs, indentation and lines. |
| <% pyapi('gws.lib.uom') %> | Units of measure: pixels, millimeters, scales and resolutions. |
| <% pyapi('gws.lib.watcher') %> | File system watcher used by the monitor. |
| <% pyapi('gws.lib.xmlx') %> | XML parsing, building and serialization, with namespace handling. |
| <% pyapi('gws.lib.zipx') %> | Creating and extracting zip archives. |

`gws.lib.vendor` contains third-party code that is not installed with pip, for example `jump` (templates), `slon` (the configuration notation), `dog` (documentation) and `umsgpack` (msgpack).

## Plugins ::

<% pyapi('gws.plugin', 'gws.plugin') %> contains the optional components. Each plugin package implements one or more extension types, for example layer types, database and OWS providers, authentication methods, model fields, templates or client tools. A plugin is loaded only if the configuration uses one of its types. External plugins, listed in the manifest, work the same way, see [](/dev-en/server/plugins).
