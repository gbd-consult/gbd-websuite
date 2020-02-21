Styling
=======

Vector feature styles can be customized via CSS. GBD WebSuite supports standard CSS properties for SVG (for example, ``fill``) and a few custom properties, which must be prefixed with ``--`` in your CSS.

TABLE
``--with-labels``~``all`` to render all labels, ``none`` to skip labels
``--label-fill``~foreground color for labels
``--label-font-size``~label font size
``--label-stroke-width``~label stroke color
``--label-stroke``~label stroke color
``--label-offset-x``~label offset from the automatic position
``--label-offset-y``~label offset from the automatic position
``--label-min-resolution``~min resolution to display labels
``--label-max-resolution``~max resolution to display labels

``--mark``~type of the end of line or edge marker (``circle``, ``square``, ``cross``, ``arrow`` )
``--mark-fill``~fill color for the marker
``--mark-size``~marker size
``--mark-stroke-width``~marker stroke color
``--mark-stroke``~marker stroke color

``--point-size``~point size for Point features
/TABLE

Styling vector layers
---------------------

For vector layers, like ``postgres`` layers, you can place your css rules directly in the configuration file under ``style``::

    "layers": [
        ...
        {
            "title": "My Vector layer",
            "type": "sql",
            ...
            "style": {
                "type": "css",
                "text": "stroke: rgb(0, 255, 255); stroke-width: 2px; fill: rgba(0, 100, 100, 0.2); --label-fill: rgb(255, 0, 0)"
            }
        }

Alternatively, you can include the rules in your project-related CSS file (see :doc:`projects`), and use the ``cssSelector`` style type ::

    // In your css file

    .myVectorLayer {
        stroke: rgb(0, 255, 255);
        stroke-width: 2px;
        fill: rgba(0, 100, 100, 0.2);
        --label-fill: rgb(255, 0, 0);
    }

    // In your config file

    {
        "title": "My Vector layer",
        "type": "postgres",
        ...
        "style": {
            "type": "cssSelector",
            "text": ".myVectorLayer"
        }
    }

Styling built-in features
-------------------------

You can customize styles for built-in features, like search results markers or measurements. There are following predefined CSS selectors:

TABLE
``.gws .modMarkerFeature``~search results marker
``.gws .modAnnotateFeature``~measured/annotated geometry
/TABLE


Styling dimensions
------------------

The dimensioning plugin uses these CSS selectors:

TABLE
``.gws .modDimensionDimLine``~main dimension line
``.gws .modDimensionDimPlumb``~a "plumb" line from the end of the main line to the edge of the object
``.gws .modDimensionDimCross``~a cross at the end of the main line
``.gws .modDimensionDimArrow``~an arrow at the end of the main line
``.gws .modDimensionDimLabel``~dimension label
/TABLE
