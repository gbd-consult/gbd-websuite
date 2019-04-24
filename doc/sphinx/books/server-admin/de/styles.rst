Styling
=======

Die Vektor-Feature-Styles können über die CSS-Datei angepasst werden. Die GBD WebSuite unterstützt standardmäßige CSS-Eigenschaften für SVG-Icons wie zum Beispiel `` fill``. Benutzerdefinierte Eigenschaften müssen in Ihrer CSS-Datei mit dem Ausdruck `` --`` zu Beginn versehen werden.

TABLE
``--label-background`` ~ Hintergrundfarbe für Beschriftungen von Features
``--label-fill`` ~ Vordergrundfarbe für Beschriftungen
``--label-font-size`` ~ Label Schriftgröße
``--label-offset-x`` ~ Label-Offset von der automatischen Position
``--label-offset-y`` ~ Label-Offset von der automatischen Position
``--label-min-resolution`` - Minimale Auflösung für die Anzeige von Beschriftungen
``--label-max-resolution`` ~ maximale Auflösung für die Anzeige von Beschriftungen

`` --mark`` ~ Ende der Linie oder der Randmarkierung (`` circle``)
`` --mark-fill`` ~ Füllfarbe für den Marker
`` --mark-size`` ~ Markergröße
`` --mark-stroke-width`` ~Breite der Markierungsstrichfarbe
`` --mark-stroke`` ~ Markierungsstrichfarbe

``--point-size`` ~ Punktgröße für Point-features
/TABLE

Styling von Vektor-Layern
-------------------------

Für Vektorebenen wie `` sql`` oder `` geojson``-Layer können Sie Ihre CSS-Regeln direkt in der Konfigurationsdatei unter `` style`` platzieren:

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

Alternativ können Sie die Regeln in Ihre projektzugehörige CSS-Datei mit aufnehmen (siehe: doc: `projects`) und den Text ``text`` für den CSS-Selektor festlegen:

    // In Ihrer CSS-Datei

    .myVectorLayer {
        stroke: rgb(0, 255, 255);
        stroke-width: 2px;
        fill: rgba(0, 100, 100, 0.2);
        --label-fill: rgb(255, 0, 0);
    }

    // In Ihrer Konfigurationsdatei

    {
        "title": "My Vector layer",
        "type": "sql",
        ...
        "style": {
            "type": "css",
            "text": ".myVectorLayer"
        }
    }

Styling von integrierten Features
---------------------------------

Sie können Stile für integrierte Features, z. B. Markierungen von Suchergebnissen oder Messungen, anpassen. Es gibt folgende vordefinierte CSS-Befehlen:

TABLE
`` .gws .modMarkerFeature`` ~ Marker für Suchergebnisse
`` .gws .modAnnotatePoint`` ~ Punktmessung
`` .gws .modAnnotateLine`` ~ Linienmessung
`` .gws .modAnnotatePolygon`` ~ Polygonmessung
`` .gws .modAnnotateBox`` ~ Boxmessung
`` .gws .modAnnotateCircle`` ~ Kreismessung
/TABLE


Styling des Dimensionierungs-Plugin
-----------------------------------

Das Dimensionierungs-Plugin verwendet folgende CSS-Befehle:

TABLE
`` .gws .modDimensionDimLine`` ~ Hauptmaßlinie. Für Hauptlinien der Dimension unterstützt die Eigenschaft `` --marker`` die zusätzlichen Werte `` cross`` und `` arrow``.
`` .gws .modDimensionDimPlumb`` ~ eine "senkrechte" Linie vom Ende der Hauptlinie bis zum Rand des Objekts
`` .gws .modDimensionDimCross`` ~ ein Kreuz am Ende der Hauptlinie
`` .gws .modDimensionDimArrow`` ~ einen Pfeil am Ende der Hauptlinie
`` .gws .modDimensionDimLabel`` ~ Dimensionsbezeichnung
/TABLE
