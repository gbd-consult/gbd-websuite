Vorlagen
========

Vorlagen werden an verschiedenen Stellen in GWS genutzt.

- für dynamische Web-Pages (s. ^web)
- bei Projekten (s. ^project) für Project Infoboxen
- bei Layern  für Layer Infoboxen und Feature-Elemente (s. ^layer und ^feature)
- bei Suchprovidern für Feature Elemente (s. ^search)
- bei OWS Diensten für ``GetFeatureInfo`` XML Snippets (s. ^ows)

Eine HTML-Vorlage ist eine Text oder HTML-Datei mit externen Variablen, die in Klammern ``{...}`` eingeschlossen sind. Zusätzlich gibt es grundlegende Programmierkonstrukte (Bedingungen, Schleifen, Funktionen), mit denen die Vorlagenausgabe in Abhängigkeit von den Variablen geändert werden kann.

Für Projekt-, Layer- und Feature-Vorlagen stellt das System Objekte ``project``, ``user`` und ggf. ``layer`` und ``feature`` mit ihren jeweiligen Eigenschaften zur Verfügung, die beim Templating verwendet werden können. Bei Feature-Vorlagen sind zusätzlich alle Attributen des Features als Variablen verfügbar.

Konfiguration
-------------

^REF gws.types.ext.template.Config

Eine Vorlagen-Konfiguration enthält zwei grundlegende Eigenschaften: den Typ (``type``) und das Subjekt (``subject``), das angibt, für welchen Zweck die Vorlage verwendet wird.

Den Ihnalt der Vorlage können Sie direkt in der Konfiguration mit ``text`` angeben, oder auf eine externe Datei mit ``path`` verweisen.

Vorlagen-Sprache
----------------

Neben den Variablen, können in Vorlagen auch Basis Programmierkonstrukte verwendet werden. Die Konstrukte sind mit dem ``@`` Zeichen markiert. Die wichtigsten davon sind:

- Bedingungen ``@if...``, ``@elif`` und ``@else``
- Schleifen ``@each``
- Funktion (Snippet) Definitionen ``@def``
- Inklusion ``@include`` mit der die Vorlagen ineinander inkludiert werden können

^SEE Eine vollständige Beschreibung aller verfügbaren Konstrukte finden Sie in der Dokumentation zur Template-Engine unter https://github.com/gebrkn/chartreux.

Vorlage Typen
-------------

text
~~~~

Text-Vorlagen enthalten Befehle der Vorlagen-Sprache, ansonsten werden alle Zeichen wortwörtlich interpretiert

html
~~~~

HTML-Vorlagen können die Befehle der Vorlagen-Sprache enthalten, beliebige HTML Formatierung und spezielle HTML Tags die unter ^print beschrieben sind.

xml
~~~

XML Vorlagen werden intern für OWS Diensten benutzt.

Beispiele
---------

Format-Vorlage eines Layers: ::

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

Formatierung eines "city" Feature, welches die Attribute "name", "area" und "population" besitzt: ::

    @if population > 100000
        <div class="big-city">{name}</div>
    @else
        <div class="small-city">{name}</div>
    @end

    <p> <strong>Area:</strong> {area} </p>
    <p> <strong>Population:</strong> {population} </p>
