# Vorlagen :/admin-de/config-az/vorlagen

Vorlagen werden an verschiedenen Stellen in der GBD WebSuite genutzt.

- für dynamische Web-Pages oder *Assets* , siehe [Web-Server](/admin-de/config-az/web)
- bei Projekten (siehe [Projekte](/admin-de/config-az/projekte)) für Project Infoboxen
- bei Layern  für Layer Infoboxen und Features, siehe [Layer](/admin-de/config-az/layer) und [Feature](/admin-de/config-az/feature)
- bei Suchprovidern für Features, siehe [Suche](/admin-de/config-az/suche)
- bei OWS Diensten für XML Dokumente, siehe [OWS](/admin-de/config-az/ows)

Eine Vorlage ist eine Text oder eine HTML-Datei mit externen Variablen, die in Klammern ``{...}`` eingeschlossen sind. Zusätzlich gibt es grundlegende Programmierkonstrukte (Bedingungen, Schleifen, Funktionen), mit denen die Vorlagenausgabe in Abhängigkeit von den Variablen geändert werden kann.

Für Projekt-, Layer- und Feature-Vorlagen stellt das System Objekte ``project``, ``user`` und ggf. ``layer`` und ``feature`` mit ihren jeweiligen Eigenschaften zur Verfügung, die beim Templating verwendet werden können. Bei Feature-Vorlagen sind zusätzlich alle Attributen des Features als Variablen verfügbar.

## Konfiguration

%reference_de 'gws.ext.config.template'

Eine Vorlagen-Konfiguration enthält zwei grundlegende Eigenschaften: den Typ (``type``) und das Subjekt (``subject``), das angibt, für welchen Zweck die Vorlage verwendet wird.

Den Inhalt der Vorlage können Sie direkt in der Konfiguration mit ``text`` angeben, oder auf eine externe Datei mit ``path`` verweisen.

## Vorlagen-Sprache

Neben den Variablen, können in Vorlagen auch Basis Programmierkonstrukte verwendet werden. Die Konstrukte sind mit dem ``@`` Zeichen markiert. Die wichtigsten davon sind:

- Bedingungen ``@if...``, ``@elif`` und ``@else``
- Schleifen ``@each``
- Funktion (Snippet) Definitionen ``@def``
- Inklusion ``@include`` mit der die Vorlagen ineinander inkludiert werden können

%info
 Eine vollständige Beschreibung aller verfügbaren Konstrukte finden Sie in der Dokumentation zur Template-Engine unter https://github.com/gebrkn/chartreux.
%end

## Vorlage Typen

### text

Text-Vorlagen enthalten Befehle der Vorlagen-Sprache, ansonsten werden alle Zeichen wortwörtlich interpretiert

### html

HTML-Vorlagen können die Befehle der Vorlagen-Sprache enthalten, beliebige HTML Formatierung und spezielle HTML Tags die unter [Drucken](/admin-de/config-az/drucken) beschrieben sind.

Zum Beispiel, hier ist eine Vorlage fur die Layer-Beschreibung (``subject: layer.description``):

```html

    <h1>{layer.title}</h1>

    <p>{layer.meta.abstract}</p>

    @if layer.has_legend
        <img src="_?cmd=mapHttpGetLegend&layerUid={layer.uid}"/>
    @end

    <ul>
        @each layer.meta.keywords as keyword
            <li>{keyword}</li>
        @end
    </ul>
```

Beschreibung (``subject: feature.description``) eines "city" Feature, welches die Attribute "name", "area" und "population" besitzt:

```html

    @if population > 100000
        <div class="big-city">{name}</div>
    @else
        <div class="small-city">{name}</div>
    @end

    <p> <strong>Area:</strong> {area} </p>
    <p> <strong>Population:</strong> {population} </p>
```

%info
 Das erste Zeichen (ausgenommen Whitespace) der Ausgabe einer HTML-Vorlage muss ``<`` sein, ansonsten wird die Vorlage als ``text`` interpretiert.
%end

### xml

XML Vorlagen werden intern für OWS Diensten benutzt. In XML Vorlagen werden zusätzliche Vorlagen-Befehle unterstützt, ``@t`` und ``@tag`` die entsprechend finale bzw container XML-Tags erzeugen. Sie https://github.com/gbd-consult/gbd-websuite/blob/master/app/gws/ext/ows/service/wms/templates/getCapabilities.1.3.cx für ein Beispiel der XML Vorlage für die GetCapabilties Operation.