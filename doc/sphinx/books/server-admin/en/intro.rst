Introduction
============

GBD WebSuite is an application webserver with emphasis on geodata processing.

GBD WebSuite as a web server:

- can serve static and templated content
- supports multi-site configurations, url routing and rewriting
- supports various authorization mechanisms (file system, database, LDAP) and fine-grained permissions

GBD WebSuite as a geo server:

- combines different sources (WMS, tile servers, databases) into a unified map
- has direct support for QGIS projects
- caches, reprojects and scales raster data as necessary
- can process and render vector data (PostGIS, shapefile, json)
- provides OGC conformant services (WMS, WMTS, WFS)

GBD WebSuite as an application server:

- provides a framework for domain-specific extensions
- has a pluggable architecture for easy integration

GBD WebSuite is proudly built upon open source software, to name a few:

- `QGIS <https://qgis.org>`_
- `MapProxy <https://mapproxy.org/>`_
- `uWSGI <https://github.com/unbit/uwsgi>`_
- `NGINX <https://www.nginx.com/>`_


and it's free and open source itself.
