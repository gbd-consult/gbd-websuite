# Server Konfiguration :/admin-de/config/server

%reference_de 'gws.server.core.Config'

## Loggen

Das Loggen erfolgt per Default in Standard-Ausgabe (*stdout*), sodass Logs mit ``docker logs`` bzw ``--log-driver`` Option abgefangen werden können (s. https://docs.docker.com/config/containers/logging/ für mehr Info). Alternativ können Sie GWS Logs in eine Datei weiterleiten.

Für das Loggen sind folgende Stufen definiert:


| TAG | EIGENSCHAFT |
|---|---|
| ``ERROR`` | schwerwiegende Fehler, die die Funktion des GWS Servers beeinträchtigen |
| ``INFO`` | zusätzliche Informationsmeldungen (Defaultwert) |
| ``DEBUG`` | zusätzliche Ausgaben zur Fehlersuche |

## Module

Die GBD WebSuite betreibt intern mehrere Servermodule:

- das ``web``-Modul, das eingehende Anfragen entgegennimmt und versendet [Web Server](/admin-de/config/web)
- das ``mapproxy`` Modul, das den gebündelten MapProxy ausführt und sich um externe Quellen, Caching und Umprojizieren kümmert
- das ``qgis`` Modul, das den gebündelten QGIS Server betreibt und QGIS Projekte und Layer rendert [QGIS](/admin-de/intro/concepts)
- das ``spool`` Modul, das den Druck und andere Hintergrundaufgaben übernimmt.
- das ``monitor`` Modul, das das Dateisystem überwacht, und bei den Änderungen in Ihren Daten einen Hot-Reload des Servers durchführt

Jedes Modul kann mit ``enabled: false`` deaktiviert werden, wenn es nicht benötigt wird (z.B. wenn Sie keine QGIS-Projekte verwenden, brauchen Sie den QGIS-Server nicht auszuführen). Sie können auch die Anzahl der *Prozessen* (``worker``) konfigurieren, die jedes Modul verwenden darf.

Für Hochlast-Workflows ist es auch möglich, verschiedene Module auf verschiedenen physikalischen Maschinen zu betreiben. Beispielsweise können Sie eine GWS-Installation einrichten, die nur das Mapproxy-Modul, eine weitere für den QGIS-Server und eine weitere für das Frontend-Web-Modul ausführt. In diesem Fall können Sie für Mapproxy und QGIS in der Web-Konfiguration ``host`` and ``port`` angeben, so dass diese über das Netzwerk abgefragt werden können.

## Monitor

Normalerweise, überwacht der Montior folgende Daten und Ordner:

- App-Konfigurationsdatei
- Projekt-Konfigurationsdateien
- Projekt-Konfigurations-Ordner, falls Sie die Option ``projectDirs`` verwenden [Applikation](/admin-de/config/applikation)
- Vorlagen
- QGIS Projekte

In der Konfiguration des Monitors können Sie bestimmte Pfade mit der ``ignore`` Option ausschließen.



## Datenablage

Im GBD WebSuite Client besteht die Möglichkeit, bestimmte Objekte, wie Markierungen, Bemaßungen oder Auswahllisten abzuspeichern und später aufzurufen. Serverseitig wird dies mit der Funktion *Datenablage* (``storage``) unterstützt. Die Datenablage wird mit dem ``storage`` Helper (siehe [Helper](/admin-de/config/helper)) konfiguriert. Zusätzlich zu der Helper-Konfiguration muss auch die ``storage`` Aktion und die Client Elemente ``Storage.Read`` und ``Storage.Write`` aktiviert werden.

Die Ablage wird in *Kategorien* (``category``) unterteilt wobei jede Kategorie einer Client-Funktion entspricht. In jeder Kategorie kann unbegrenzte Anzahl von Einträgen gespeichert werden. Aktuell sind folgende Kategorien implementiert:

| OPTION | BEDEUTUNG |
|---|---|
| ``Alkis`` | Flurstückslisten, siehe [ALKIS](/admin-de/plugin/alkis) |
| ``Annotate`` | vom Benutzer erstellte Markierungen |
| ``Dimension`` | Bemaßungen, siehe [Feature Bemaßungen](/admin-de/plugin/dimension) |
| ``Select`` | Auswahllisten |
| ``Styles`` | vom Benutzer editierte Style Eigenschaften |

### Helper ``storage``

%reference_de 'gws.base.storage.manager.Config'

In der Konfiguration des Helpers geben Sie an, welche User-Rollen den Zugriff zu bestimmten Ablagen-Kategorien  haben. Zu jeder Kategorie kann eine Liste von Regeln zugeordnet werden, die angeben welche Rollen die Einträge in dieser Kategorie erzeugen (``write``) oder lesen (``read``) kann, oder beides (``all``). Außerdem können Sie ein Sternchen (``*``) eingeben, das für alle Kategorien steht. Im folgenden Beispiel haben die Rollen ``nutzer`` und ``expert`` Lesezugriff auf alle Kategorien, und die Rolle ``expert`` Schreibzugriff auf ``Dimension``:

```javascript

"helpers": [
    ...
    {
        "type": "storage",
        "permissions": [
            {
                "category": "*",
                "mode": "read",
                "access": [
                    { "role": "nutzer", "type": "allow"},
                    { "role": "expert", "type": "allow"}
                ]
            },
            {
                "category": "Dimension",
                "mode": "write",
                "access": [
                    { "role": "expert", "type": "allow"}
                ]
            }
        ]
    }
    ...
]
```
