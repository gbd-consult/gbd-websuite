# Einfaches Projekt :/admin-de/einfaches-projekt

<!--
    Schreibstil:
    Aussagen über die GBD WebSuite als kurze, bündige Fakten.
        z.B.  Die GBD WebSuite wird als Container ausgeliefert. 
    Handlungsaufforderungen als imperativ, förmliche Anrede
        z.B.  Erstellen Sie eine Datei mit dem Namen `docker-compose.yml`:
    
    Zielsetzung:
    - Aufbauend auf "Guide: Schnellstart" und "Guide: Konfigurationsgrundlagen"
    - Führen des Lesers durch Erstellung aller Dateien in /data die benötigt 
      werden um ein lauffähiges Projekt mit Karte zu erstellen.
    - Kurze Erklärungen von Hintergründen wenn dies ohne große Abschweifungen möglich ist.
    - Ansonsten viele Links auf relevante Bereiche der Konfiguration in Texte einarbeiten.
    - Davon ausgehen, dass Leser eigenständig auf Links klicken kann also kein
        "wenn Sie mehr wissen wollen klicken sie [hier]"
    -->

Dieser Guide führt den Leser durch die Erstellung einer GBD WebSuite Konfiguration,
unter Berücksichtung aller unbedingt benötigten Einstellungen um eine Karte darzustellen.

Es wird auf dem [Schnellstart](/admin-de/schnellstart) Guide aufgebaut.  
Empfohlene vorbereitende Lektüre: [Konfigurationsgrundlagen](/admin-de/konfigurationsgrundlagen) 


## Verzeichnisstruktur

Am Ende des [Schnellstart](/admin-de/schnellstart) Guides haben Sie in einem
von Ihnen gewählten Verzeichnis eine Datei `docker-compose.yml` und die beiden 
Verzeichnisse `data` sowie `gws-var` erstellt.

In diesem Guide ist das `data`-Verzeichnis relevant. Für die GBD WebSuite ist 
dieses Verzeichnis unter `/data` erreichbar, daher wird in diesem Guide auch 
stets dieser Name für das Verzeichnis verwendet. Auch wenn es auf Ihrem Computer
an einem anderen Pfad liegt, referenzieren Sie Pfade zu Dateien innerhalb der Konfiguration stets aus Sicht der WebSuite, 
unter Verwendung eines absoluten Pfades, beginnend mit `/data`.

In dem noch leeren `/data` Verzeichnis erstellen Sie folgende 
Unterverzeichnisse:

- `/data/web` für statische Assets, wie z.B. ein Logo oder ein Stylesheet
- `/data/assets` für dynamische Assets
- `/data/config` für globale Konfigurationsdateien
- `/data/config/projects` für Projektkonfigurationen
- `/data/qgis` für QGIS Projekte

## Einstiegspunkt

Die erste Konfigurationsdatei die von der GBD WebSuite gelesen wird ist `/data/config.cx`.

Erstellen Sie diese mit folgendem Inhalt:

```title="/data/config.cx"
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
In der ersten Zeile wird Anwendern genereller, lesender Zugriff auf alle
Komponenten der WebSuite gestattet. Dies ist für diesen Guide nötig, da das
Thema [Benutzerkonten und Berechtigungen](/admin-de/themen/auth) hier nicht
näher behandelt wird.

Danach werden drei *actions* ([web](),[map](),[project]()) aktiviert.
Actions sind serverseitige Funktionsgruppen.

Als nächstes werden die Dateien `web.cx` und `client.cx` inkludiert, gefolgt von 
einer Liste namens `projects` die aktuell nur einen Eintrag hat: den Inhalt der 
Datei `myproject.cx`.

Erstellen Sie die drei referenzierten Dateien. Lassen Sie diese zunächst leer. 
Existieren inkludierte Dateien nicht schlägt der Startvorgang der GBD WebSuite fehl.


## WebServer, WebSites & Rewrites

In der Datei `/data/config/web.cx` werden alle Einstellungen hinterlegt die es 
der GBD WebSuite ermöglichen Dateien auszuliefern.

Dateien die im nach `root` angegebenen Verzeichnis liegen werden unter 
`http://localhost:3333/dateiname.endung` direkt ausgeliefert.

Dateien die im Verzeichnis `assets` liegen werden vom Server als Templates 
betrachtet. Hier kann Logik in der Datei hinterlegt werden: Beispielsweise kann eine 
Datei in `assets`, basierend auf der Information ob ein zugreifender Nutzer 
eingeloggt ist, eine Liste von für den Benutzer erlaubten Projekten ausliefern 
oder alternativ ein Login-Formular.

`host` wird gegen die aufrufende Domain geprüft um die Anfrage einer "site" zuzuweisen. 
Zeigen zwei Domains (z.B. example.com und mywebgis.de) auf den Server auf 
dem die GBD WebSuite läuft, können die Domains unterschiedlich behandelt werden.

%info

Das Verhalten des `host` Felds entspricht dem der `server_name` Variable des [nginx Webservers](https://nginx.org/en/docs/http/request_processing.html).

%end

Die `rewrite` Regeln erlauben unterschiedliche Struktur für den auf die Domain 
folgenden Teil der URL. Sie folgenden Regeln für [Python RegEx](https://docs.python.org/3/library/re.html#regular-expression-syntax)

Übernehmen Sie zunächst die folgenden Regeln:

```title="/data/config/web.cx"
web.sites+ {
    root.dir "/data/web"
    assets.dir "/data/assets"
    host "*"

    rewrite+ {
        pattern "^/$"
        target "/_/webAsset/path/index.cx.html"
    }
    rewrite+ {
        pattern "^/project/([a-zA-Z0-9_.-]+)"
        target "/_/webAsset/projectUid/$1/path/project.cx.html"
    }
}
```

Nach einem Neustart der GBD WebSuite können im Verzeichnis `/data/web` abgelegte 
Dateien im Browser abgerufen werden.

Testen Sie dies indem Sie die Datei `/data/web/test.html` mit dem Inhalt
`<h1>Test erfolgreich</h1>` erstellen, und diese im Browser über die URL
`http://localhost:3333/test.html` öffnen.


## Benötigte Assets

Es muss eine Startseite existieren die unter `http://localhost:3333/` bereit
gestellt wird.

Die erste `rewrite` Regel sorgt dafür, dass diese URL den Inhalt der Datei
`/data/assets/index.cx.html` anzeigt.

Erstellen Sie diese Datei und hinterlegen den folgenden Inhalt:

```html title="/data/assets/index.cx.html"
<h1>Meine GBD WebSuite Seite</h1>
<h2>Projekte:</h2>
<ul>
@for prj in projects
    <li>
        <a href="/project/{prj.uid}">{prj.title | html}</a>
        <p>{prj.metadata.abstract | html}</p>
    </li>
@end
</ul>
```

Rufen Sie die URL für die Startseite im Browser auf(`http://localhost:3333/`). Die Überschrift sollte angezeigt werden. Die Logik zur Darstellung einer Liste von Projekten ist in der Datei bereits mit hinterlegt. Noch existiert aber kein Projekt und die Liste ist leer.

%info

Wird noch die Seite aus aus dem Schnellstart Guide angezeigt, drücken Sie Umschalt+F5 um Ihren Browser zu zwingen die Seite wirklich neuzuladen.

%end

Vor der Erstellung des ersten Projektes muss noch ein weiteres Asset erstellt 
werden: `/data/assets/project.cx.html` bindet die 
clientseitige Applikation ein, die einen interaktiven Umgang mit der Karte ermöglicht.

```html title="/data/assets/project.cx.html"
<!DOCTYPE html>
<html>
<head>
    <title>{project.title}</title>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0"/>
    <link rel="stylesheet" href="/_/webSystemAsset/path/light.css" type="text/css"/>
    <link rel="stylesheet" href="/stylesheet.css">
</head>
<body>
    <script id="gwsOptions" type="application/json">
        {
            "projectUid": "{project.uid}"
        }
    </script>

    <script src="/_/webSystemAsset/path/vendor.js"></script>
    <script src="/_/webSystemAsset/path/app.js"></script>
</body>
</html>
```

Übernehmen Sie den Inhalt dieser Datei exakt, er bedarf keiner Anpassungen.

Ergänzen Sie ebenfalls die Datei `/data/web/stylesheet.css`

```css title="/data/web/stylesheet.css"
.gws {
	position: fixed;
	left: 0;
	top: 0;
	right: 0;
	bottom: 0;
}
```



## Projekte

Es ist möglich mehrere Projekte in der GBD WebSuite zu konfigurieren. 
Jedes Projekt kann unterschiedliche Features der GBD WebSuite aktivieren oder 
deaktivieren, und kann eine Karte und eine Übersichtskarte enthalten.

Sie haben bereits eine Datei angelegt in der die Projektkonfiguration 
hinterlegt werden soll: `/data/config/projects/myproject.cx`

Ergänzen Sie den folgenden Inhalt in der Projektkonfiguration:
```title="/data/config/projects/myproject.cx"
{
    uid myproject
    title "Mein Projekt"
    metadata.abstract "Dies ist mein erstes GBD WebSuite Projekt"

    map.crs 3857
    map.center [757072,6663486]
    map.layers+ {
        type tile
        title "OSM"
        provider.url "https://osmtiles.gbd-consult.de/ows/{{z}}/{{x}}/{{y}}.png"
    }
}
```


Stellen Sie sicher, dass jedes Projekt eine einzigartige `uid` erhält. Sie wird 
in der URL zur Karte verwendet. Verwenden Sie nach Möglichkeit nur Klein- und 
Großbuchstaben, Ziffern sowie Unter- und Bindestriche.

`title` sowie `metadata.abstract` können Sie frei verwenden.

Der `map.layers+` Block fügt eine Layer zur Karte hinzu. Lesen Sie dazu [Maps und Layers](TODO LINK ZU THEMA)

Laden Sie die Seite im Browser neu. Auf der Startseite wird nun das Projekt gelistet.
Klicken Sie auf den Link des Projektes um zur Karte zu gelangen.


## Client

Aktuell wird die Karte ohne jegliche [Steuerelemente](TODO LINK) angezeigt.

Sie haben zu Beginn die Datei `/data/config/client.cx` erstellt.
Dort soll eine Auswahl an Steuerelementen für den [Client](TODO LINK) 
getroffen werden die für alle Projekte gilt.

Ergänzen Sie folgenden Inhalt:

```title="/data/config/client.cx"
client.elements [
    { tag "Infobar.ZoomOut" }
    { tag "Infobar.ZoomIn" }
    { tag "Infobar.ZoomReset" }
    { tag "Infobar.Position" }
    { tag "Infobar.Scale" }
    { tag "Infobar.Loader" }
    { tag "Infobar.Spacer" }
    { tag "Infobar.Link" }
    { tag "Infobar.LinkButton" }
    { tag "Infobar.HomeLink" }
    { tag "Infobar.Help" }
    { tag "Infobar.About" }
]
```

Diese Liste fügt die Steuerelemente der Infobar am unteren Kartenrand in der 
angegebenen Reihenfolge hinzu.
