# QField :/admin-de/themen/qfield

[QField](https://qfield.org) ist eine mobile GIS-Anwendung, die auf QGIS basiert und die offline und online Erfassung sowie Bearbeitung von Geodaten im Außendienst ermöglicht. Die GBD WebSuite emuliert die QFieldCloud-API, sodass QField direkt mit dem GBD WebSuite Server synchronisieren kann – ohne einen separaten QFieldCloud-Dienst.

%info
Die GBD WebSuite implementiert die QFieldCloud REST-API selbst. QField kommuniziert
dabei direkt mit dem GBD WebSuite Server und benötigt keinen externen QFieldCloud-Dienst.
%end

**Bereitstellung von Daten für QField**: Die GBD WebSuite liest das konfigurierte QGIS-Projekt, sucht darin alle PostgreSQL-Layer die im QGIS-Projekt mit QFieldSync als "offline editierbar" markiert wurden, lädt deren Daten und erzeugt daraus ein GeoPackage-Paket. Hintergrundkarten werden als Raster-Kacheln gerendert und ebenfalls ins Paket aufgenommen. Das QGIS-Projektfile wird so angepasst, dass es auf die GeoPackage-Layer verweist.

**Synchronisation QField Daten mit GBD WebSuite**: QField sendet die vorgenommenen Änderungen als sogenannte "Deltas" an die GBD WebSuite. Diese extrahiert die geänderten Features und schreibt sie über die konfigurierten Datenmodelle in die PostgreSQL-Datenbank zurück.

## QGIS-Projekt vorbereiten :qgis-projekt

Das QGIS-Projekt muss mit dem QGIS Plugin **QFieldSync** für QField vorbereitet werden,
bevor es in der GBD WebSuite konfiguriert werden kann.

In QFieldSync wird für jeden Layer eine Aktion festgelegt:

| Aktion         | Bedeutung                                                      |
|----------------|----------------------------------------------------------------|
| `offline`      | Layer wird als GeoPackage ins Paket aufgenommen und ist editierbar |
| `remove`       | Layer wird aus dem mobilen Paket entfernt                      |
| `no_action`    | Layer bleibt unverändert (z. B. WMS-Hintergrundkarten)        |

Zusätzlich können in QFieldSync folgende Projekteigenschaften gesetzt werden:

- **Hintergrundkarte** (`baseMapType`): Aus einem Kartenthema (`mapTheme`) oder einem einzelnen Layer (`singleLayer`) gerenderte Raster-Kacheln
- **Bereich** (`areaOfInterest`): WKT-Geometrie, die den Ausschnitt für Daten und Kacheln begrenzt
- **Medienverzeichnisse** (`attachmentDirs`): Verzeichnisse mit Anhängen wie Fotos

%info
Das QGIS-Projekt muss auf dem GBD WebSuite Server erreichbar sein. Pfade zu
Datenbankverbindungen sollten über eine `pg_service.conf` konfiguriert sein, damit
sowohl QGIS Desktop als auch der GBD WebSuite Server dieselbe Verbindungskonfiguration nutzen können. Siehe dazu auch das Thema [PostgreSQL](/admin-de/themen/postgresql/pg_service.conf).
%end

## Konfiguration :konfiguration

Das QField-Plugin wird als Action vom Typ `qfieldcloud` in der GBD WebSuite Konfigurationsdatei eingebunden:

```javascript
actions+ {
    type "qfieldcloud"

    projects+ {
        uid "mein_projekt"
        title "Mein QField-Projekt"
        provider.path "/data/projects/mein_projekt.qgs"
        mapCacheLifeTime "7d"
        access { role "user" type "allow" }
    }
}
```

**Konfigurationsparameter**

| Parameter           | Typ        | Beschreibung                                                  |
|---------------------|------------|---------------------------------------------------------------|
| `uid`               | `str`      | Eindeutiger Bezeichner des Projekts                           |
| `title`             | `str`      | Anzeigename des Projekts in der QField-App                    |
| `provider.path`     | `str`      | Dateipfad zum QGIS-Projektfile auf dem Server                 |
| `mapCacheLifeTime`  | `Duration` | Gültigkeitsdauer des Karten-Cache (z. B. `7d`, `0` = deaktiviert) |
| `access`            |            | Zugriffssteuerung (siehe [Zugriffssteuerung](/admin-de/themen/qfield/zugriffssteuerung)) |
| `models`            | Liste      | Optionale Datenmodelle für editierbare Layer                  |

**Demo-Konfiguration**

Die folgende Konfiguration entspricht der Demo [Editieren mit QField](https://docs.gbd-websuite.de/demo/qfield_demo):

```javascript
uid "qfield_demo"
title "Editieren mit QField"

actions+ {
    type "qfieldcloud"
    access "allow all"
    projects+ {
        uid "qfield_demo"
        title "QField Demo"
        provider.path "/demos/poi_districts_qfield.qgs"
        mapCacheLifeTime "7d"
        access "allow all"
    }
}

map.layers+ {
    title "Points of interest"
    type "qgisflat"
    provider.path "/demos/poi_districts_qfield.qgs"
    sourceLayers.names [ "Points of interest" ]
}
```

**Datenmodelle**

Für jeden editierbaren PostgreSQL-Layer kann ein Datenmodell konfiguriert werden.
Das Modell steuert, welche Felder synchronisiert werden, und ermöglicht
Datenbankzugriff mit Rechteverwaltung.

Ohne explizites Modell erstellt die GBD WebSuite automatisch ein generisches Modell,
das alle Felder und alle Features umfasst.

**Modell konfigurieren**

```javascript
actions+ {
    type "qfieldcloud"

    projects+ {
        uid "mein_projekt"
        title "Mein QField-Projekt"
        provider.path "/data/projects/mein_projekt.qgs"

        models+ {
            type "postgres"
            tableName "public.begehungen"
            uid "begehungen"
            permissions.read  { role "user" type "allow" }
            permissions.edit  { role "user" type "allow" }
        }

        models+ {
            type "postgres"
            tableName "public.schaeden"
            uid "schaeden"
            permissions.read  { role "user" type "allow" }
            permissions.edit  { role "user" type "allow" }
        }
    }
}
```

%info
Der `tableName` im Modell muss mit dem Namen der PostgreSQL-Tabelle übereinstimmen,
auf die der entsprechende QGIS-Layer zeigt.
%end

**Datei-Uploads** :datei-upload

QField erlaubt es, Fotos und andere Dateien direkt an Features anzuhängen.
Damit diese Anhänge in der Datenbank gespeichert werden, wird im Modell ein
virtuelles Feld vom Typ `file` konfiguriert:

```javascript
models+ {
    type "postgres"
    tableName "public.begehungen"
    uid "begehungen"

    fields+ {
        type "file"
        name "foto"
        contentColumn "foto_inhalt"
        pathColumn    "foto_pfad"
        nameColumn    "foto_name"
    }
}
```

| Parameter       | Beschreibung                                        |
|-----------------|-----------------------------------------------------|
| `contentColumn` | Datenbankspalte, in der der Dateiinhalt (Bytes) gespeichert wird |
| `pathColumn`    | Datenbankspalte für den Dateipfad                   |
| `nameColumn`    | Datenbankspalte für den originalen Dateinamen (optional) |

QField überträgt Anhänge in zwei Schritten: Zuerst wird der Dateipfad mit den
Feature-Änderungen übermittelt, danach wird der Dateiinhalt in einer separaten
Anfrage hochgeladen. Die GBD WebSuite ordnet Dateiinhalt und Feature anhand des
`pathColumn`-Wertes einander zu.

## Zugriffssteuerung :zugriffssteuerung

Sowohl die Action selbst als auch die einzelnen Projekte unterstützen
GBD WebSuite Zugriffsregeln. Dadurch kann gesteuert werden, welche Benutzer
welche Projekte sehen und bearbeiten dürfen:

```javascript
actions+ {
    type "qfieldcloud"
    access { role "admin" type "allow" }

    projects+ {
        uid "projekt_intern"
        title "Internes Projekt"
        provider.path "/data/intern.qgs"
        access { role "user" type "allow" }

        models+ {
            type "postgres"
            tableName "public.daten"
            permissions.read  { role "user"     type "allow" }
            permissions.edit  { role "editor"   type "allow" }
        }
    }
}
```

## QField App einrichten :qfield-app

In der QField App wird der GBD WebSuite Server als QFieldCloud-Server eingetragen:

1. QField öffnen und zu **QFieldCloud-Projekte** navigieren
2. Server-URL eintragen: `https://mein-server.example.com/qfc`
   (entspricht dem in GBD WebSuite konfigurierten Pfad der `qfieldcloud`-Action)
3. Mit Benutzername und Passwort des GBD WebSuite Benutzers anmelden
4. Das gewünschte Projekt aus der Projektliste herunterladen

%info
Der URL-Pfad des QFieldCloud-Endpunkts ergibt sich aus der GBD WebSuite
Action-Konfiguration. Standardmäßig ist dies `/qfc`. Der genaue Pfad kann über
die `actionName`-Einstellung angepasst werden.
%end

## Hintergrundkarten-Cache :hintergrundkarten-cache

Hintergrundkarten werden als Raster-Kacheln vom QGIS Server gerendert und auf
dem GBD WebSuite Server gecacht. Der Parameter `mapCacheLifeTime` steuert,
wie lange zwischengespeicherte Kacheln als gültig gelten:

| Wert  | Bedeutung                                     |
|-------|-----------------------------------------------|
| `0`   | Kein Cache – Kacheln werden immer neu gerendert |
| `1d`  | Cache für einen Tag gültig                    |
| `7d`  | Cache für sieben Tage gültig (empfohlen)      |

Der Cache wird automatisch invalidiert, wenn sich das QGIS-Projektfile ändert.

Cache-Dateien liegen unter:
```
{VAR_DIR}/qfieldcloud/projects/{project_uid}/cache/
```

## CLI-Befehl :cli-befehl

Pakete können auch manuell über die Kommandozeile erstellt werden, z. B. zu
Testzwecken:

```bash
gws qfieldcloudPackage --projectUid=mein_gws_projekt --qfcProjectUid=mein_projekt --dir=/tmp/paket
```

| Parameter       | Beschreibung                                        |
|-----------------|-----------------------------------------------------|
| `projectUid`    | UID des GBD WebSuite Projekts                       |
| `qfcProjectUid` | UID des QField-Projekts aus der Action-Konfiguration |
| `dir`           | Ausgabeverzeichnis für das erstellte Paket          |
