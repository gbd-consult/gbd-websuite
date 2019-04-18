Printing
========

A GBD WebSuite project can provide multiple print templates. For each template, you can also configure a list of DPI for each quality grade ("Normal", "Optimal" and "High"). Note that high-DPI printing requires lots of memory and can't even be possible with sources that impose restrictions on request bounding boxes. Printing an A3 map with DPI 1200 will probably not work.

Draft printing
--------------

The GBD WebSuite client also supports the "Draft" ("screenshot") print mode, which doesn't perform map rendering, but instead prints the on-screen map as a bitmap image.


Template types
--------------

html
~~~~

A html print template is an html template which contains the following helper tags:

TABLE
``<gws:page width="297" height="210" margin="5 5 5 5"/>`` ~ page size and margins (in mm)
``<gws:map width="150" height="150"/>`` ~ will be replaced by the generated map image
/TABLE


qgis
~~~~

A QGIS template is a print composition from a QGIS project file. If there are multiple compositions, you can use ``compositionIndex`` or ``compositionName`` to identify the composition to use. Since maps in GWS are rendered separately, both template and map backgrounds in the composition must be set to ``None`` (transparent).
