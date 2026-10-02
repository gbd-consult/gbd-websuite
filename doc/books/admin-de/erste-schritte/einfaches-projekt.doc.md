# Einfaches Projekt :/admin-de/erste-schritte/einfaches-projekt

Dieser Leitfaden führt durch alle Dateien, die ein lauffähiges Projekt mit einer Karte benötigt. Er baut auf [](/admin-de/erste-schritte/schnellstart) und [](/admin-de/erste-schritte/konfigurationsgrundlagen) auf.

## Verzeichnisstruktur

Nach dem Schnellstart haben Sie ein Verzeichnis mit `docker-compose.yml` sowie den Ordnern `data` und `gws-var`. Aus Sicht der WebSuite ist `data` unter `/data` erreichbar; Pfade in der Konfiguration geben Sie stets absolut aus dieser Sicht an (beginnend mit `/data`).

Legen Sie in `/data` folgende Unterverzeichnisse an:

- `/data/web` – statische Assets (Logo, Stylesheet …)
- `/data/config` – globale Konfigurationsdateien
- `/data/config/projects` – Projektkonfigurationen
- `/data/qgis` – QGIS-Projekte

## Einstiegspunkt

Die erste gelesene Datei ist `/data/config.cx`:

```javascript title="/data/config.cx"
{
    permissions.read "allow all"

    actions [
        { type web }
        { type map }
        { type project }
    ]

    @include /data/config/web.cx
    @include /data/config/client.cx

    projects [
        @include /data/config/projects/myproject.cx
    ]
}
```

Die erste Zeile gewährt allgemeinen Lesezugriff – ausreichend für diesen Leitfaden, der [Benutzerkonten und Berechtigungen](/admin-de/themen/zugriff/authentifizierung) nicht behandelt. Danach werden drei Server-Aktionen aktiviert und die Dateien `web.cx` und `client.cx` sowie ein Projekt eingebunden.

Erstellen Sie die drei referenzierten Dateien schon jetzt (zunächst leer) – fehlt eine eingebundene Datei, schlägt der Start fehl.

## Webserver

`/data/config/web.cx` steuert das Ausliefern von Dateien:

```javascript title="/data/config/web.cx"
web.sites+ {
    root.dir "/data/web"
    host "*"
}
```

Dateien unter `root` werden unverändert ausgeliefert; `host` wird gegen die aufrufende Domain geprüft. Beide Werte entsprechen der Voreinstellung — die Datei ist hier vor allem der Ort, an dem Sie später TLS, Zugriffsregeln oder eigene Adressen ergänzen.

Nach einem Neustart sind Dateien aus `/data/web` abrufbar. Zum Test: `/data/web/test.html` mit Inhalt `<h1>Test erfolgreich</h1>` anlegen und `http://localhost:3333/test.html` öffnen.

## Startseite und Projektseite

Um Seiten müssen Sie sich nicht kümmern. Die WebSuite beantwortet zwei Adressen von sich aus:

| Adresse | Inhalt |
|---|---|
| `/` | Startseite mit der Liste aller Projekte, die der Nutzer sehen darf |
| `/project/<uid>` | die Karte des Projekts im Client |

Dahinter stehen zwei mitgelieferte Vorlagen und zwei Rewrite-Regeln, die jede Webseite automatisch erhält. Eigene Regeln brauchen Sie dafür nicht.

Ein eigenes Aussehen genügt in den meisten Fällen. Legen Sie dafür `/data/web/style.css` an — diesen Dateinamen bindet die WebSuite von sich aus in beide Seiten ein:

```css title="/data/web/style.css"
html, body {
    height: 100%;
    margin: 0;
}

.gws {
    position: fixed;
    left: 0; top: 0; right: 0; bottom: 0;
}
```

Die Regel für `.gws` ist nicht bloß Kosmetik: Der Kartencontainer bekommt seine Größe erst dadurch. Ohne sie bleibt die Projektseite leer, obwohl alles richtig geladen wird.

%info
Reichen die mitgelieferten Seiten nicht, ersetzen Sie sie durch eigene Vorlagen mit den Subjects `application.home` und `project.home`. Die [](/admin-de/themen/darstellung/templates) beschreibt das, und wer stattdessen eigene Adressen vergeben will, findet die nötigen Regeln unter [](/admin-de/konfiguration/web).
%end

## Projekt

Ergänzen Sie `/data/config/projects/myproject.cx`:

```javascript title="/data/config/projects/myproject.cx"
{
    uid myproject
    title "Mein Projekt"
    metadata.abstract "Dies ist mein erstes GBD WebSuite Projekt"

    map.crs 3857
    map.center [757072, 6663486]
    map.layers+ {
        type tile
        title "OSM"
        provider.url "https://osmtiles.gbd-consult.de/ows/{{z}}/{{x}}/{{y}}.png"
    }
}
```

Jedes Projekt braucht eine eindeutige `uid` – sie erscheint in der URL. Die eingebaute Adressregel erkennt nur Kleinbuchstaben, Ziffern, Unter- und Bindestriche; wer Großbuchstaben oder Punkte verwenden will, muss eine eigene Regel hinterlegen. Der Block `map.layers+` fügt einen [](/admin-de/themen/karten/layer) hinzu. Nach dem Neuladen erscheint das Projekt auf der Startseite und führt per Klick zur Karte.

## Client

Die Karte wird noch ohne Bedienelemente angezeigt. In `/data/config/client.cx` wählen Sie projektübergreifend die [Client-Elemente](/admin-de/themen/darstellung/client):

```javascript title="/data/config/client.cx"
client.elements [
    { tag "Infobar.ZoomOut" }
    { tag "Infobar.ZoomIn" }
    { tag "Infobar.ZoomReset" }
    { tag "Infobar.Position" }
    { tag "Infobar.Scale" }
    { tag "Infobar.Loader" }
    { tag "Infobar.HomeLink" }
    { tag "Infobar.Help" }
    { tag "Infobar.About" }
]
```

Diese Liste fügt die Bedienelemente der Infobar am unteren Kartenrand in dieser Reihenfolge hinzu.

%info
Damit steht ein einfaches Projekt. Vertiefende Konfigurationsmöglichkeiten finden Sie im Kapitel [](/admin-de/themen).
%end
