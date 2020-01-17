Styling
=======

Vektor-Feature-Stile können über CSS angepasst werden. Die GBD WebSuite unterstützt standard CSS-Eigenschaften für SVG Symbole (zum Beispiel, ``fill``) und einige benutzerdefinierte Eigenschaften, welche in Ihrer CSS-Konfiguration mit ``--`` vorangestellt werden müssen.

TABLE
``--label-background``~background color for feature labels
``--label-fill``~foreground color for labels
``--label-font-size``~label font size
``--label-offset-x``~label offset from the automatic position
``--label-offset-y``~label offset from the automatic position
``--label-min-resolution``~min resolution to display labels
``--label-max-resolution``~max resolution to display labels

``--mark``~end of line or edge marker (``circle``)
``--mark-fill``~fill color for the marker
``--mark-size``~marker size
``--mark-stroke-width``~marker stroke color
``--mark-stroke``~marker stroke color

``--point-size``~point size for Point features
/TABLE

Styling von Vektorlayern
------------------------

Für `` sql`` oder `` geojson``-Vektorlayer können Sie Ihre CSS-Regeln direkt in der Konfigurationsdatei unter `` style`` einfügen. ::


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

Alternativ können Sie die Regeln in Ihre projektbezogene CSS-Datei mit aufnehmen (siehe: doc: `projects`) und ``text`` bei den CSS-Selektor setzen ::

    // In Ihrer CSS-Datei

    .myVectorLayer {
        stroke: rgb(0, 255, 255);
        stroke-width: 2px;
        fill: rgba(0, 100, 100, 0.2);
        --label-fill: rgb(255, 0, 0);
    }

    // In Ihrer config-Datei

    {
        "title": "My Vector layer",
        "type": "sql",
        ...
        "style": {
            "type": "css",
            "text": ".myVectorLayer"
        }
    }

Styling der eingebauten Funktionen
----------------------------------

Sie können den Stil der integrierten Funktionen anpassen, z. B. Markierungen von Suchergebnisse oder Messungen. Es gibt folgende vordefinierte CSS-Selektoren:

TABLE
``.gws .modMarkerFeature``~search results marker
``.gws .modAnnotatePoint``~point measure
``.gws .modAnnotateLine``~line measure
``.gws .modAnnotatePolygon``~polygon measure
``.gws .modAnnotateBox``~box measure
``.gws .modAnnotateCircle``~circle measure
/TABLE


Styling des Bemaßungs-Plugins
-----------------------------

Das Bemaßungs-Plugin verwendet diese CSS-Selektoren: 

TABLE
``.gws .modDimensionDimLine``~main dimension line. For dimension main lines, the ``--marker`` property supports additional values ``cross`` and ``arrow``.
``.gws .modDimensionDimPlumb``~a "plumb" line from the end of the main line to the edge of the object
``.gws .modDimensionDimCross``~a cross at the end of the main line
``.gws .modDimensionDimArrow``~an arrow at the end of the main line
``.gws .modDimensionDimLabel``~dimension label
/TABLE
