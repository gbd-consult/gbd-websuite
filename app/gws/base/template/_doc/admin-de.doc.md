# Vorlagen :/admin-de/config/template
========

Vorlagen werden an verschiedenen Stellen in der GBD WebSuite genutzt.

- für dynamische Web-Pages oder *Assets* (s. ^web)
- bei Projekten (s. ^project) für Project Infoboxen
- bei Layern  für Layer Infoboxen und Features (s. ^layer und ^feature)
- bei Suchprovidern für Features (s. ^search)
- bei OWS Diensten für XML Dokumente (s. ^ows)

Eine Vorlage ist eine Text oder eine HTML-Datei mit externen Variablen, die in Klammern `{...}` eingeschlossen sind. Zusätzlich gibt es grundlegende Programmierkonstrukte (Bedingungen, Schleifen, Funktionen), mit denen die Vorlagenausgabe in Abhängigkeit von den Variablen geändert werden kann.

Für Projekt-, Layer- und Feature-Vorlagen stellt das System Objekte `project`, `user` und ggf. `layer` und `feature` mit ihren jeweiligen Eigenschaften zur Verfügung, die beim Templating verwendet werden können. Bei Feature-Vorlagen sind zusätzlich alle Attributen des Features als Variablen verfügbar.

Konfiguration
-------------

^REF gws.types.ext.template.Config

Eine Vorlagen-Konfiguration enthält zwei grundlegende Eigenschaften: den Typ (`type`) und das Subjekt (`subject`), das angibt, für welchen Zweck die Vorlage verwendet wird.

Den Inhalt der Vorlage können Sie direkt in der Konfiguration mit `text` angeben, oder auf eine externe Datei mit `path` verweisen.

Vorlagen-Sprache
----------------

Neben den Variablen, können in Vorlagen auch Basis Programmierkonstrukte verwendet werden. Die Konstrukte sind mit dem `@` Zeichen markiert. Die wichtigsten davon sind:

- Bedingungen `@if...`, `@elif` und `@else`
- Schleifen `@each`
- Funktion (Snippet) Definitionen `@def`
- Inklusion `@include` mit der die Vorlagen ineinander inkludiert werden können

^SEE Eine vollständige Beschreibung aller verfügbaren Konstrukte finden Sie in der Dokumentation zur Template-Engine unter https://github.com/gebrkn/chartreux.

## Vorlage Typen

### :/admin-de/config/template/type/*



