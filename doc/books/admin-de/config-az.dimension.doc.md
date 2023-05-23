# Bemaßung :/admin-de/config-az/dimension

%reference_de 'gws.plugin.dimension.Config'

Mit der Funktion Bemaßung (``dimension``) kann der Nutzer im Client Bemaßungen zeichnen, die optional an bestimmte Vektor-Objekte einrasten.

Um diese Funktion zu nutzen, aktivieren Sie die Aktion ``dimension``. Wenn Sie das Einrasten benötigen, legen Sie einen Vektor-Layer an (z.B. ``postgres``) und fügen Sie dessen ``uid`` unter ``layers`` ein. Ein Beispiel:

```javascript

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
```

Mit dieser Konfiguration werden Bemaßungen an Konturen aus der Tabelle ``strassen`` einrasten, mit einer Toleranz von 20 Pixel.

Zusätzlich müssen auch Client-Elemente ``Sidebar.Dimension`` und ``Toolbar.Dimension`` aktiviert werden, siehe [Client](/admin-de/config-az/client).

Wenn die Bemaßungen gespeichert werden sollen, muss auch die Datenablage freigeschaltet und konfiguriert werden, siehe [Datenablage](/admin-de/config-az/storage).

## CSS Anpassungen

Sie können das Aussehen von Bemaßungen (z.B. Farbe, Linienbreite) mit diesen CSS-Selektoren anpassen:

| OPTION | BEDEUTUNG |
|--|--|
| ``.gws.modDimensionDimLine`` | Hauptlinie, normalerweise mit der ``--marker`` Eigenschaft ``cross`` oder ``arrow`` |
| ``.gws.modDimensionDimPlumb`` | Lotlinie |
| ``.gws.modDimensionDimCross`` | ``cross`` (Kreuz) Marker |
| ``.gws.modDimensionDimArrow`` | ``arrow`` (Pfeil) Marker |
| ``.gws.modDimensionDimLabel`` | Beschriftung |
