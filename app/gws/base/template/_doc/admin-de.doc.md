# Vorlagen :/admin-de/config/template

Vorlagen werden an verschiedenen Stellen in der GBD WebSuite genutzt.

- für dynamische Web-Pages oder *Assets* , siehe [Web-Server](/admin-de/config/web)
- bei Projekten (siehe [Projekte](/admin-de/config/projekte)) für Project Infoboxen
- bei Layern  für Layer Infoboxen und Features, siehe [Layer](/admin-de/config/layer) und [Feature](/admin-de/config/feature)
- bei Suchprovidern für Features, siehe [Suche](/admin-de/config/suche)
- bei OWS Diensten für XML Dokumente, siehe [OWS](/admin-de/plugin/ows)

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

## Vorlage Typen :/admin-de/config/template/type

### text

Text-Vorlagen enthalten Befehle der Vorlagen-Sprache, ansonsten werden alle Zeichen wortwörtlich interpretiert

### :/admin-de/config/template/type/html

### xml

XML Vorlagen werden intern für OWS Diensten benutzt. In XML Vorlagen werden zusätzliche Vorlagen-Befehle unterstützt, ``@t`` und ``@tag`` die entsprechend finale bzw container XML-Tags erzeugen. Sie https://github.com/gbd-consult/gbd-websuite/blob/master/app/gws/ext/ows/service/wms/templates/getCapabilities.1.3.cx für ein Beispiel der XML Vorlage für die GetCapabilties Operation.

### :/admin-de/config/template/type/map

### :/admin-de/config/template/type/py
