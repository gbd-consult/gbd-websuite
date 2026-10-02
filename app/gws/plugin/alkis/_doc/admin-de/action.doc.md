# Aktion "alkis" :/admin-de/konfiguration/action/alkis

Die Aktion `alkis` stellt das Backend für die Flurstückssuche bereit und ermöglicht die Suche in ALKIS-Katasterdaten sowie die Anzeige von Flurstücks-Details, Eigentümer- und Buchungsinformationen. Mit `dbUid` und `dataSchema` binden Sie die ALKIS-Datenbank an, mit `eigentuemer` und `buchung` steuern Sie den geschützten Zugriff auf Eigentümer- und Grundbuchdaten, und mit `exporters` konfigurieren Sie den Datenexport.

Voraussetzungen, Indizierung und die Bedienung im Client gehören nicht hierher, sondern zum [Fachthema ALKIS](/admin-de/themen/fachmodule/alkis).

## Buchungs- und Eigentümerdaten

Flurstücksdaten sind in der Regel unkritisch; Buchungs- und vor allem Eigentümerdaten unterliegen dagegen dem Datenschutz. Beide Bereiche werden deshalb getrennt konfiguriert (`buchung`, `eigentuemer`) und tragen jeweils eigene `permissions`. Ohne Freigabe bleiben sie unzugänglich, auch wenn die Aktion selbst erreichbar ist.

## Kontrollmodus

Für Eigentümerdaten lässt sich zusätzlich ein *Kontrollmodus* aktivieren. Ist `controlMode` gesetzt, muss der Nutzer bei jeder Abfrage einen Grund angeben – etwa ein Aktenzeichen. Die Eingabe wird gegen `controlRules` geprüft, eine Liste regulärer Ausdrücke; trifft keiner zu, wird die Abfrage abgelehnt.

```javascript
eigentuemer {
    permissions.read "allow sachbearbeiter, deny all"
    controlMode true
    controlRules [
        "^[A-Z]{2}-\\d{4}/\\d{2}$"
    ]
    logTable "public.alkis_log"
}
```

## Protokolltabelle

Mit `logTable` benennen Sie eine Tabelle, in die jeder Zugriff auf Eigentümerdaten geschrieben wird – auch die abgelehnten. Protokolliert werden Zeitpunkt, IP-Adresse, Anmeldename und Anzeigename des Nutzers, die Kontrolleingabe, das Ergebnis der Prüfung sowie Anzahl und Kennungen der abgefragten Flurstücke.

%warn
Die Tabelle wird **nicht** automatisch angelegt. Fehlt sie, schlägt das Protokollieren fehl. Legen Sie sie vor der Inbetriebnahme selbst an.
%end

```sql
CREATE TABLE alkis_log (
    id SERIAL PRIMARY KEY,
    app_name VARCHAR(255),
    date_time TIMESTAMP,
    ip VARCHAR(255),
    login VARCHAR(255),
    user_name VARCHAR(255),
    control_input VARCHAR(255),
    control_result INTEGER,
    fs_count INTEGER,
    fs_ids TEXT
)
```

Der Datenbank-Nutzer benötigt auf dieser Tabelle nur das Recht `INSERT`, nicht `SELECT`. So kann die Anwendung protokollieren, die Protokolle aber nicht selbst auslesen.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "alkis"
    dbUid "DB_ALKIS"
    dataSchema "public"
    indexSchema "gws82"
    crs 25832
    limit 200

    ui {
        useSelect true
        useExport true
        searchSpatial true
    }

    eigentuemer {
        permissions.read "allow sachbearbeiter, deny all"
        logTable "public.alkis_log"
    }
    buchung {
        permissions.read "allow sachbearbeiter, deny all"
    }

    exporters+ {
        type "geojson"
        title "Basisinformationen (JSON)"
        models+ {
            fields [
                { type "text"  name "fs_flurstueckskennzeichen" title "Flurstückskennzeichen" }
                { type "text"  name "fs_recs_gemarkung_text" title "Gemarkung" }
                { type "float" name "fs_recs_amtlicheFlaeche" title "Fläche" }
            ]
        }
    }
}
```

`dbUid`, `dataSchema`, `indexSchema` und `crs` binden die ALKIS-Datenbank und den Suchindex an. `limit` begrenzt die Trefferzahl, `ui` schaltet Bedien-Funktionen im Client frei (hier Auswahl, Export und räumliche Suche). Der Zugriff auf die schützenswerten Bereiche wird getrennt freigegeben: `eigentuemer` und `buchung` tragen jeweils eigene `permissions.read` und bleiben ohne Freigabe unzugänglich; `logTable` protokolliert die Eigentümer-Zugriffe. Über `exporters` definieren Sie Export-Konfigurationen, deren `models` die auszugebenden Felder anhand der flachen ALKIS-Schlüssel festlegen.

%ref "gws.plugin.alkis.action.Config"
%demo "alkis_export"
%demo "alkis_mini"
%demo "alkis_ui"
