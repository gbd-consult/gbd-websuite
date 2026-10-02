# Vorlage "html" :/admin-de/konfiguration/template/html

Die Vorlage `html` verarbeitet Vorlagen in der Jump-Sprache und erzeugt daraus HTML sowie, für den Druck, auch PDF- und Bild-Ausgaben. Den Vorlageninhalt geben Sie entweder direkt über `text` oder als Datei über `path` an. Zusätzliche Befehle wie `@map` und `@legend` binden Karte und Legende in die Ausgabe ein.

Gibt eine Vorlage ausdrücklich ein Antwort-Objekt zurück, wird der erzeugte Text verworfen und stattdessen dieses Objekt ausgeliefert. Die übergebenen Argumente erreichen Sie über das Objekt `_ARGS`.

Die Syntax der Jump-Sprache – Variablen, Bedingungen, Schleifen und das Einbinden von Dateien – ist im Projekt [Jump](https://github.com/gebrkn/jump) beschrieben. Dieselbe Sprache verwenden die [Konfigurationsgrundlagen](/admin-de/erste-schritte/konfigurationsgrundlagen) für das Aufteilen der Konfiguration.

## Druck

Für die PDF-Ausgabe erweitert die WebSuite die Jump-Sprache um eigene Befehle. Sie bestimmen das Seitenformat und die Stellen, an denen Karte, Legende sowie Kopf- und Fußzeilen eingesetzt werden.

### Seite

`@page` legt das Format der gedruckten Seite fest. Alle Maße sind Millimeter.

```
@page (
    width="210"
    height="297"
    margin="20"
)
```

### Karte

`@map` rendert die aktuelle Karte an dieser Stelle. Erforderlich sind `width` und `height`; die übrigen Angaben überschreiben den Ausschnitt, den der Nutzer im Client gewählt hat.

```
@map (
    width="180"
    height="120"
)
```

| Parameter | Bedeutung |
|---|---|
| `width`, `height` | Größe des Kartenbildes in mm |
| `bbox` | fester Ausschnitt in Einheiten der Projektion |
| `center` | fester Mittelpunkt in Einheiten der Projektion |
| `scale` | fester Maßstab |
| `rotation` | Drehung in Grad |

### Legende

`@legend` fügt die Legende ein. Ohne Angabe erscheinen die Legenden aller sichtbaren Layer; mit `layers` schränken Sie die Auswahl auf bestimmte Layer-uids ein, getrennt durch Leerzeichen.

```
@legend (
    layers="strassen gebaeude"
)
```

### Kopf- und Fußzeilen

`@header` und `@footer` sind Blockbefehle und definieren Kopf- und Fußzeile für den mehrseitigen Druck. Beide sind eigenständige Untervorlagen: Sie erhalten dieselben Argumente wie die Hauptvorlage und zusätzlich `page` (aktuelle Seite) und `numpages` (Gesamtzahl der Seiten) – damit lässt sich eine Seitennummerierung erzeugen.

```
@header
    <div class="kopf">Übersichtskarte</div>
@end header

@footer
    <div class="fuss">Seite {page} von {numpages}</div>
@end footer
```

`@pagebreak` erzwingt einen Seitenumbruch.

## Beispiel-Konfiguration ::

```javascript
templates+ {
    subject "feature.title"
    type "html"
    text "<b>{{{{name}}}}</b>"
}
templates+ {
    subject "feature.teaser"
    type "html"
    path "/data/templates/poi_teaser.cx.html"
}
```

Das `subject` legt den Zweck einer Vorlage fest: `feature.title` erzeugt die Überschrift eines Features, `feature.teaser` seine Kurzansicht. Kurze Vorlagen geben Sie direkt über `text` an, umfangreichere über `path` als Datei.

%info
Die Konfiguration wird selbst mit der Jump-Sprache verarbeitet. Eine `{`, auf die kein Leerzeichen folgt, deutet der Server daher schon beim Einlesen der Konfiguration als Beginn eines Ausdrucks. Eine über `text` eingebettete Vorlage enthält aber ihrerseits Jump-Platzhalter wie `{{name}}`. Damit diese nicht bereits beim Einlesen ausgewertet werden, verdoppeln Sie die Klammern: `{{{{name}}}}` wird beim Einlesen zu `{{name}}`, das die Vorlage anschließend gegen das Feature auflöst. Geben Sie die Vorlage über `path` als eigene Datei an, entfällt das Verdoppeln – die Datei durchläuft die Konfigurationsverarbeitung nicht und verwendet einfache Klammern (`{{name}}`).
%end

%ref "gws.plugin.template.html.Config"
