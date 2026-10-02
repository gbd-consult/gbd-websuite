# CSS-Stile :/admin-de/konfiguration/style

Das Erscheinungsbild von Vektorobjekten – Füllung, Linien, Marker und Beschriftungen – beschreiben Sie mit CSS. Die GBD WebSuite versteht dabei einen Teil der Standard-CSS-Eigenschaften und ergänzt sie um **eigene Eigenschaften**, die es in CSS nicht gibt, etwa für Marker und Beschriftungen.

Eigene Eigenschaften werden mit zwei Bindestrichen geschrieben (`--marker`, `--label-fill`). Das ist die CSS-Schreibweise für benutzerdefinierte Eigenschaften; ein CSS-Parser oder ein Editor stört sich dadurch nicht daran. Die WebSuite entfernt das Präfix beim Einlesen, `--marker` und `marker` sind also gleichbedeutend.

%warn
Unbekannte Eigenschaften werden nicht stillschweigend übergangen: Die WebSuite meldet `style: invalid css property` im Log und ignoriert die Angabe. Prüfen Sie bei unerwarteter Darstellung zuerst das Log.
%end

## Ein Stil konfigurieren

Ein Stil hat zwei Formen. Bevorzugt verweisen Sie mit `cssSelector` auf eine Regel aus einer eigenen CSS-Datei, die über das `web`-Verzeichnis ausgeliefert wird:

```javascript
style {
    cssSelector ".myLayer"
}
```

So halten Sie die Gestaltung getrennt von der Konfiguration, und mehrere Layer teilen sich dasselbe Aussehen, ohne dass Sie die Regeln wiederholen. Alternativ geben Sie die Regeln unmittelbar als Text an:

```javascript
style {
    text "fill: rgba(255,0,0,0.4); stroke: red; stroke-width: 2"
}
```

Einige Objekte gestaltet der Client über einen fest vergebenen Selektor. So zeichnet er Treffer aus Suche und Objektabfrage mit der Klasse `.modMarkerFeature`; eine gleichnamige Regel in Ihrer CSS-Datei ändert deren Hervorhebung.

Für die **Druckausgabe** kann dieselbe CSS-Datei verwendet werden. Binden Sie sie in der Druckvorlage ein, dann sehen Karte und PDF gleich aus und Sie pflegen die Gestaltung an einer Stelle:

```html title="print.cx.html"
<link rel="stylesheet" type="text/css" href="style.css">
```

%info
Die Druckausgabe wird aus dem Dateisystem heraus erzeugt, nicht über den Webserver. `href` muss daher eine lokale Datei bezeichnen – entweder relativ zur Vorlage, also `style.css` für eine Datei im selben Verzeichnis, oder als absoluter Pfad wie `/data/web/style.css`.
%end

## Geometrie und Beschriftung

Ein Stil beschreibt beides zugleich: die Geometrie und die Beschriftung des Objekts. Über `--with-geometry` und `--with-label` schalten Sie eines von beiden ab – etwa um nur Beschriftungen ohne sichtbare Geometrie darzustellen.

| Eigenschaft | Werte | Bedeutung |
|---|---|---|
| `--with-geometry` | `all`, `none` | Geometrie zeichnen. Voreinstellung `all` |
| `--with-label` | `all`, `none` | Beschriftung zeichnen. Voreinstellung `all` |

## Flächen und Linien

Diese Eigenschaften tragen ihre Standard-CSS-Namen und werden **ohne** Präfix geschrieben.

| Eigenschaft | Werte | Bedeutung |
|---|---|---|
| `fill` | Farbe | Füllfarbe der Fläche |
| `stroke` | Farbe | Linienfarbe |
| `stroke-width` | Zahl (px) | Linienbreite. Voreinstellung `0`, die Linie bleibt also unsichtbar |
| `stroke-dasharray` | Liste von Zahlen | Strichmuster, z. B. `5 2` |
| `stroke-dashoffset` | Zahl | Versatz des Strichmusters |
| `stroke-linecap` | `butt`, `round`, `square` | Linienenden. Voreinstellung `butt` |
| `stroke-linejoin` | `bevel`, `round`, `miter` | Eckenform. Voreinstellung `miter` |
| `stroke-miterlimit` | Zahl | Grenzwert für spitze Ecken |

Für Punktgeometrien bestimmt `--point-size` (Voreinstellung `10`) den Durchmesser, `--icon` bindet stattdessen eine Grafik ein. Mit `--offset-x` und `--offset-y` verschieben Sie das gezeichnete Objekt gegenüber seiner tatsächlichen Lage.

## Marker

Ein *Marker* ist ein Symbol, das an den Stützpunkten einer Geometrie gezeichnet wird. Ohne `--marker` wird kein Symbol dargestellt.

| Eigenschaft | Werte | Bedeutung |
|---|---|---|
| `--marker` | `circle`, `square`, `arrow`, `cross` | Form des Symbols |
| `--marker-size` | Zahl (px) | Größe |
| `--marker-fill` | Farbe | Füllfarbe |
| `--marker-stroke` | Farbe | Linienfarbe |
| `--marker-stroke-width` | Zahl | Linienbreite |

Zusätzlich gelten `--marker-stroke-dasharray`, `--marker-stroke-dashoffset`, `--marker-stroke-linecap`, `--marker-stroke-linejoin` und `--marker-stroke-miterlimit` mit derselben Bedeutung wie bei der Geometrie.

## Beschriftung

Den Text einer Beschriftung erzeugt eine [Vorlage](/admin-de/themen/darstellung/templates) mit dem Subject `feature.label`; die folgenden Eigenschaften bestimmen sein Aussehen.

| Eigenschaft | Werte | Bedeutung |
|---|---|---|
| `--label-fill` | Farbe | Textfarbe |
| `--label-font-family` | Schriftname | Voreinstellung `sans-serif` |
| `--label-font-size` | Zahl (px) | Voreinstellung `12` |
| `--label-font-style` | `normal`, `italic` | |
| `--label-font-weight` | `normal`, `bold` | |
| `--label-line-height` | Zahl | Zeilenabstand bei mehrzeiligem Text |
| `--label-align` | `left`, `right`, `center` | Ausrichtung. Voreinstellung `center` |
| `--label-placement` | `start`, `end`, `middle` | Lage entlang der Geometrie. Voreinstellung `middle` |
| `--label-offset-x`, `--label-offset-y` | Zahl | Versatz gegenüber der Ankerposition |
| `--label-background` | Farbe | Hintergrundfläche hinter dem Text |
| `--label-padding` | Liste von Zahlen | Innenabstand der Hintergrundfläche |
| `--label-min-scale`, `--label-max-scale` | Zahl | Maßstabsbereich, in dem die Beschriftung erscheint |

Für die Kontur des Textes gelten `--label-stroke`, `--label-stroke-width` sowie die übrigen `--label-stroke-*`-Eigenschaften analog zur Geometrie.

Mit `--label-min-scale` und `--label-max-scale` blenden Sie Beschriftungen in ungeeigneten Maßstäben aus. Voreingestellt ist der gesamte Bereich, jede Beschriftung erscheint also in jedem Maßstab.

## Beispiel-Konfiguration ::

```javascript
style {
    text """
        fill: rgba(0, 100, 200, 0.3);
        stroke: rgb(0, 60, 140);
        stroke-width: 2;
        --label-fill: rgb(0, 60, 140);
        --label-font-size: 13;
        --label-font-weight: bold;
        --label-placement: middle;
    """
}
```

Dieser Stil beschreibt Geometrie und Beschriftung zugleich. `fill` mit halbdurchsichtiger Farbe füllt die Fläche, `stroke` und `stroke-width` zeichnen eine 2 Pixel breite Kontur. Die `--label-*`-Eigenschaften gestalten den von der Vorlage `feature.label` erzeugten Text: `--label-fill` setzt die Textfarbe, `--label-font-size` und `--label-font-weight` die Schrift, `--label-placement middle` platziert die Beschriftung mittig auf der Geometrie.
