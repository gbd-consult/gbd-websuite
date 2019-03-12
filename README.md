# GBD WebSuite

Application webserver with emphasis on geodata processing.

The GBD WebSuite is a web-based open source GIS platform for geodata processing. It includes the GBD WebSuite Server and GBD WebSuite Client, which are characterized in addition to the classic WebGIS functionality by the modular and efficient integration of external applications and new functionalities, to enable extensive configurations. The core libraries of client and server are designed slim. The whole architecture is plugin based. The QGIS integration is also implemented as a plugin, which communicates natively with QGIS.

*GBD WebSuite as a web server*:

    can serve static and templated content
    supports multi-site configurations, url routing and rewriting
    supports various authorization mechanisms (file system, database, LDAP) and fine-grained permissions

*GBD WebSuite as a geo server*:

    combines different data sources (files, OGC services, databases) into a unified map
    has direct support for QGIS projects
    caches, reprojects and scales raster data as necessary
    can process and directly render vector data (PostGIS, shapefile, json)
    provides OGC conformant services (WMS, WMTS, WFS)

*GBD WebSuite as an application server*:

    provides a framework for domain-specific extensions and has a pluggable architecture for easy integration.
