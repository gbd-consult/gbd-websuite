# OWS-Dienst "wfs" :/admin-de/konfiguration/owsService/wfs

Der Dienst `wfs` stellt die durchsuchbaren Vektor-Layer eines Projekts als Web Feature Service bereit, sodass externe Anwendungen Features samt Geometrie und Attributen abrufen können. Unterstützt werden die WFS-Versionen 2.0.0, 2.0.1 und 2.0.2 nach dem Profil „Basic"; verarbeitet werden ausschließlich `GET`-Anfragen mit KVP-Kodierung.

## Vorlagen

Der Dienst unterstützt Vorlagen mit folgenden Subjects:

| Subject | Erzeugtes Dokument |
|---|---|
| `ows.GetCapabilities` | Capabilities-Dokument des Dienstes |
| `ows.GetFeature` | Feature-Antwort |
| `ows.DescribeFeatureType` | Schema-Beschreibung der Objektarten |
| `ows.GetPropertyValue` | Antwort auf eine Abfrage einzelner Eigenschaften |
| `ows.ListStoredQueries`, `ows.DescribeStoredQueries` | Auskunft über gespeicherte Abfragen |

## Beispiel-Konfiguration ::

```javascript
owsServices+ {
    type "wfs"
    supportedCrs [ 4326 25832 ]
    metadata {
        name "MEINE_STADT_WFS"
        title "WFS Meine Stadt"
        abstract "Vektordaten der Stadtverwaltung"
    }
}

map.layers+ {
    title "Stadtteile"
    type "postgres"
    tableName "edit.district"
    ows.xmlns "demo"
}

map.layers+ {
    title "Points of interest"
    type "postgres"
    tableName "edit.poi"
    ows.featureName "demo:poi"
}
```

Der Dienst stellt jeden Vektor-Layer der Projektkarte als Objektart (`FeatureType`) bereit. `supportedCrs` bestimmt, in welchen Koordinatenbezugssystemen Geometrien geliefert werden. Mit `ows.xmlns` setzen Sie das Namensraum-Präfix, unter dem die Objektart erscheint; mit `ows.featureName` überschreiben Sie den vollständigen Namen einer Objektart einzeln. In `metadata` hinterlegen Sie die im Capabilities-Dokument sichtbaren Angaben.

%ref "gws.plugin.ows_server.wfs.Config"
%demo "service_wfs_auth_demo"
%demo "service_wfs_name_demo"
%demo "service_wfs_paging_demo"
%demo "service_wfs_simple_demo"
