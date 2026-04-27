# Projekte :/admin-de/themen/projekte

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.base.project.core.Config">gws.base.project.core.Config</a>
</div>

Ein Projekt ist die zentrale Konfigurationseinheit der GBD WebSuite. Es bündelt Karte, Layer, Suche, Druckvorlagen und weitere Funktionalität zu einer eigenständigen Anwendung. Auf einer GBD WebSuite Instanz können mehrere Projekte parallel betrieben werden.

Die Projektverwaltung wird über die Aktion `project` freigeschaltet:

```javascript
{
    actions+ {
        type project
    }

    projects [
        @include /data/config/projects/meinprojekt.cx
    ]
}

```

## uid und Titel

Jedes Projekt muss eine einzigartige `uid` erhalten. Sie wird in der URL zur Karte verwendet:

```
http://localhost:3333/project/meinprojekt
```

Verwenden Sie für die `uid` ausschließlich Buchstaben, Ziffern sowie Unter- und Bindestriche. Die `uid` sollte nach der Erstveröffentlichung nicht mehr geändert werden, da sie in URLs und Lesezeichen auftauchen kann.

Mit `title` wird der Anzeigename des Projekts festgelegt, der im Client und auf der Startseite erscheint.

```javascript
{
    uid "meinprojekt"
    title "Mein Projekt"
}

```

## Metadaten

<div class="reference">
    Referenz: <a href="/doc/8.4/admin-de/reference/index.html#gws.base.metadata.core.Config">gws.base.metadata.core.Config</a>
</div>

Über `metadata` können beschreibende Informationen zum Projekt hinterlegt werden. Diese stehen in Asset-Templates zur Verfügung und können beispielsweise auf der Projektübersichtsseite angezeigt werden.

```javascript
{
    uid "meinprojekt"
    title "Mein Projekt"
    metadata.abstract "Kurzbeschreibung des Projekts"
    metadata.keywords ["Geodaten", "Stadtplan"]
}

```

Die wichtigsten Metadatenfelder:

| Feld | Beschreibung |
| --- | --- |
| `metadata.abstract` | Kurzbeschreibung des Projekts, z. B. für die Startseite. |
| `metadata.keywords` | Liste von Schlagwörtern. |
| `metadata.contactName` | Ansprechpartner für das Projekt. |
| `metadata.contactEmail` | E-Mail-Adresse des Ansprechpartners. |

## Einbindung von Projekten

Die GBD WebSuite bietet drei Möglichkeiten, Projekte in die Hauptkonfiguration einzubinden.

### projects

Mit `projects` wird eine explizite Liste von Projekten in der Hauptkonfiguration definiert. Jedes Projekt kann als direkter Block oder per `@include` eingebunden werden:

```javascript
{
    projects [
        @include /data/config/projects/projekt_a.cx
        @include /data/config/projects/projekt_b.cx
    ]
}

```
Diese Methode eignet sich, wenn eine überschaubare, fest definierte Anzahl von Projekten verwaltet wird.

### projectDirs

Mit `projectDirs` werden ein oder mehrere Verzeichnisse angegeben. Die GBD WebSuite lädt automatisch alle Projektkonfigurationen, die in diesen Verzeichnissen gefunden werden:

```javascript
{
    projectDirs ["/data/config/projects"]
}

```
Diese Methode eignet sich, wenn Projekte häufig hinzugefügt oder entfernt werden, da keine Anpassung der Hauptkonfiguration erforderlich ist.

### projectPaths

Mit `projectPaths` wird eine Liste einzelner Dateipfade zu Projektkonfigurationen angegeben:

```javascript
{
    projectPaths [
        "/data/config/projekte/projekt_a.cx"
        "/data/config/projekte/projekt_b.cx"
    ]
}

```
`projectPaths` und `projectDirs` können auch kombiniert und zusammen mit `projects` verwendet werden.

## Projektspezifische Überschreibungen

Viele Einstellungen, die global in der Hauptkonfiguration definiert sind, können auf Projektebene überschrieben oder erweitert werden. Dies ermöglicht es, eine gemeinsame Basis für alle Projekte zu pflegen und nur die Unterschiede je Projekt zu definieren.

### Assets

Mit `assets` können projektspezifische Ressourcen wie CSS-Dateien oder Bilder hinterlegt werden, die die globalen Assets ergänzen oder ersetzen:

```javascript
{
    uid "meinprojekt"
    assets.dir "/data/assets/meinprojekt"
}

```

### Aktionen

Aktionen können auf Projektebene hinzugefügt werden, um Funktionalität nur für dieses Projekt bereitzustellen:

```javascript
{
    uid "meinprojekt"
    actions+ {
        type search
    }
}

```

### Client-Einstellungen

Mit `client` können die Client-Einstellungen projektspezifisch angepasst werden. So lassen sich zum Beispiel unterschiedliche Werkzeuge für verschiedene Projekte konfigurieren:

```javascript
{
    uid "meinprojekt"
    client.elements [
        { tag "Toolbar.Measure" }
        { tag "Sidebar.Layers" }
    ]
}

```

Weitere Informationen zur Client-Konfiguration finden Sie im Thema [Client](/admin-de/config/client "Client").

<div class="admonition_info">Einen vollständigen Einstieg in die Projektkonfiguration bietet der Guide <a href="/doc/8.4/admin-de/einfaches-projekt/index.html">Einfaches Projekt</a>.

</div>
