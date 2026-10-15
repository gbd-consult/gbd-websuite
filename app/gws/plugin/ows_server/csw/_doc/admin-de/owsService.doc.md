# OWS-Dienst "csw" :/admin-de/konfiguration/owsService/csw

Der Dienst `csw` stellt Metadaten als Catalogue Service for the Web bereit, sodass externe Anwendungen den Katalog durchsuchen und einzelne Metadatensätze abrufen können. Unterstützt wird die CSW-Version 2.0.2; es ist nur eine Teilmenge des Standards implementiert. Die Metadaten werden nach ISO 19115 ausgegeben.

## Vorlagen

Der Dienst unterstützt Vorlagen mit folgenden Subjects:

| Subject | Erzeugtes Dokument |
|---|---|
| `ows.GetCapabilities` | Capabilities-Dokument des Dienstes |
| `ows.GetRecords` | Trefferliste einer Katalogabfrage |
| `ows.GetRecordById` | einzelner Metadatensatz |
| `ows.DescribeRecord` | Schema-Beschreibung der Metadatensätze |

## Beispiel-Konfiguration ::

```javascript
owsServices+ {
    type "csw"
    metadata {
        title "Metadatenkatalog Meine Stadt"
        abstract "Katalog der Geodaten der Stadtverwaltung"
        contactOrganization "Stadtverwaltung"
        contactEmail "gis@stadt.example"
    }
}
```

In den Katalog nimmt der Dienst jedes öffentliche Objekt der Konfiguration auf, etwa ein Projekt oder einen Layer, dessen `metadata` eine `catalogUid` trägt; Objekte ohne `catalogUid` erscheinen nicht. In `metadata` beschreiben Sie den Katalogdienst selbst; diese Angaben erscheinen im Capabilities-Dokument. Die Ausgabe erfolgt stets nach ISO 19115; die Option `profile` wird nicht ausgewertet.

%ref "gws.plugin.ows_server.csw.Config"
