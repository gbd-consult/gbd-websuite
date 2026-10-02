# OWS-Dienst "csw" :/admin-de/konfiguration/owsService/csw

Der Dienst `csw` stellt die Metadaten eines Projekts als Catalogue Service for the Web bereit, sodass externe Anwendungen den Katalog durchsuchen und einzelne Metadatensätze abrufen können. Unterstützt wird die CSW-Version 2.0.2; es ist nur eine Teilmenge des Standards implementiert. Die Metadaten werden nach ISO 19115 ausgegeben.

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

Der Dienst gibt die Metadaten des Projekts und seiner Layer als Katalog aus. In `metadata` beschreiben Sie den Katalogdienst selbst; diese Angaben erscheinen im Capabilities-Dokument. Die Ausgabe erfolgt standardmäßig nach ISO 19115; über die Option `profile` können Sie stattdessen das Dublin-Core-Profil (`DCMI`) wählen.

%ref "gws.plugin.ows_server.csw.Config"
