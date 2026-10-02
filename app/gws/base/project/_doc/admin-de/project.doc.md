# Projekt :/admin-de/konfiguration/project

Ein [Projekt](/admin-de/themen/grundlagen/architektur/projekte) bündelt eine Karte mit ihren Layern, die Suche, Druckvorlagen und weitere Funktionen zu einer eigenständigen Anwendung. Es ist die Einheit, die Nutzer im Client öffnen; eine Installation kann beliebig viele Projekte enthalten, die vieles aus der Applikation erben und gezielt überschreiben.

Unter `map` konfigurieren Sie die Karte des Projekts, unter `printers` die Druckvorlagen und unter `actions` die projektspezifischen Funktionen. Weitere Bestandteile sind unter anderem Suchprovider (`finders`), Datenmodelle (`models`), Info-Vorlagen (`templates`) und die Metadaten (`metadata`).

## Beispiel-Konfiguration ::

```javascript
projects+ {
    uid "stadtplan"
    title "Stadtplan"
    metadata {
        abstract "Öffentlicher Stadtplan der Stadt Musterstadt"
        keywords ["stadtplan" "poi"]
    }
    map {
        center [344371, 5677471]
        zoom.initScale 25000
        layers+ {
            title "Punkte von Interesse"
            type "qgis"
            provider.path "/data/qgis/poi.qgs"
        }
    }
    actions+ { type "map" }
    actions+ { type "search" }
}
```

Jedes Projekt braucht eine eindeutige `uid` – sie erscheint in der URL. `title` und `metadata` beschreiben das Projekt für die Projektliste und die Metadaten-Dienste. Unter `map` liegt die Karte des Projekts mit ihren Layern. Die `actions`-Liste schaltet die Funktionen dieses Projekts frei; sie **ergänzt** die global in der Applikation definierten Aktionen. Nur hier aufgeführte Aktionen stehen im Projekt zur Verfügung.

%ref "gws.base.project.core.Config"
