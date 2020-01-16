Vorlagen/Templating
===================

HTML-Vorlagen
--------------

Eine HTML-Vorlage ist eine Text- / HTML-Datei mit externen Variablen, die in `` {...} `` eingeschlossen sind. Zusätzlich gibt es grundlegende Programmierkonstrukte (Bedingungen, Schleifen, Funktionen), mit denen die Vorlagenausgabe in Abhängigkeit von den Variablen geändert werden kann.

Für Projekt-, Layer- und Feature-Vorlagen stellt das System Objekte `` project``, `` layer`` und `` feature`` mit ihren jeweiligen Eigenschaften zur Verfügung, die beim Templating verwendet werden können. Hier ist ein Beispiel für eine Formatierungsvorlage von Features. ::


    ## <- diese Zeichenkombination markiert Kommentare
    ## Formatierung eines "city" Feature, welches die folgenden Attribute besitzt: "name", "area", "population"

    @with feature.attributes as atts

        @if atts.population > 100000
            <div class="big-city">{atts.name | html}</div>
        @else
            <div class="small-city">{atts.name | html}</div>
        @end

        <p> <strong>Area:</strong> {atts.area} </p>
        <p> <strong>Population:</strong> {atts.population} </p>

    @end


Eine vollständige Beschreibung aller verfügbaren Funktionen finden Sie in der Dokumentation zur Template-Engine <https://github.com/gebrkn/chartreux> _.

Konfigurationsvorlagen
----------------------

Konfigurationsvorlagen (`` config.cx``) ähneln HTML-Vorlagen und verwenden dieselben Programmierkonstrukte. Ein wichtiger Unterschied ist, dass Variablen in * zwei * Klammern eingeschlossen werden müssen: `` {{...}} ``. Eine Konfigurationsvorlage sollte im JSON-Format vorliegen. Dann kann auch die "Shortcut" -JSON-Syntax verwendet werden (`Dokumentation <https://github.com/gebrkn/slon>` _).

Beispiel einer Konfigurationsvorlage ::


    ## Konfiguration der Hauptanwendung

    @include database-config.cx
    @include server-config.cx

    timeZone "Europe/Berlin"

    ## Wir haben vier Standorte, von denen jeder sein eigenes "root" Verzeichnis besitzt

    web {
        sites [
            @each [1, 2, 3] as siteIndex
                {
                    host "www{{siteIndex}}.mydomain.com"
                    root.dir "/data/web/{{siteIndex}}"
                    errorPage {
                        type "html"
                        path "/data/templates/error.cx.html"
                    }
                }
            @end
        ]
    }
