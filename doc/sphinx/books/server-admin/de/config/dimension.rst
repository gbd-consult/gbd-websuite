Bemaßung
========

^REF gws.ext.action.dimension.Config

Mit der Funktion Bemaßung (``dimension``) kann der Nutzer im Client Bemaßungen zeichnen, die optional an bestimmte Vektor-Objekte einrasten.

Um diese Funktion zu nutzen, aktivieren Sie die Aktion ``dimension``. Wenn Sie das Einrasten benötigen, legen Sie einen Vektor-Layer an (z.B. ``postgres``) und fügen Sie dessen ``uid`` unter ``layers`` ein. Ein Beispiel: ::

    ## in einer Projekt-Konfiguration

    "uid": "my_projekt",

    "map": {
        "layers": [
            ...
            {
                "type": "postgres",
                "table": {
                    "name": "public.strassen",
                    "geometryColumn": "kontur"
                }
            }
            ...

    "api": {
        "actions": [
            ...
            {
                "type": "dimension",
                "layers", ["my_projekt.map.strassen"],
                "pixelTolerance": 20
            }
            ...

Mit dieser Konfiguration werden Bemaßungen an Konturen aus der Tabelle ``strassen`` einrasten, mit der Toleranz von 20 Pixel.

Zusätzlich müssen auch Client-Elemente ``Sidebar.Dimension`` und ``Toolbar.Dimension`` aktiviert werden (s. ^client).

Wenn die Bemaßungen gespeichert werden sollen, muss auch die Datenablage freigeschaltet und konfiguriert werden (s. ^storage).

CSS Anpassungen
---------------

Sie können das Aussehen von Bemaßungen (z.B. Farbe, Linienbreite) mit diesen CSS-Selektoren anpassen:

{TABLE}
``.gws.modDimensionDimLine`` | Hauptlinie, normalerweise mit der ``--marker`` Eigenschaft ``cross`` oder ``arrow``
``.gws.modDimensionDimPlumb`` | Lotlinie
``.gws.modDimensionDimCross`` | ``cross`` (Kreuz) Marker
``.gws.modDimensionDimArrow`` | ``arrow`` (Pfeil) Marker
``.gws.modDimensionDimLabel`` | Beschriftung
{/TABLE}
