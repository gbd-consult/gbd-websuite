# Aktion "ows" :/admin-de/konfiguration/action/ows

Die Aktion `ows` beantwortet Anfragen an die konfigurierten OWS-Dienste (z. B. WMS, WFS, WMTS, CSW). Sie muss aktiv sein, damit die OWS-Dienste eines Projekts von außen erreichbar sind, und liefert zusätzlich die zugehörigen XML-Schemata aus.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "ows"
    permissions.read "allow all"
}

owsServices+ {
    type "csw"
    access "allow all"
    metadata {
        title "Katalogdienst"
        abstract "CSW-Dienst der WebSuite"
    }
}
```

Die Aktion selbst hat keine eigenen Optionen; sie macht die konfigurierten Dienste von außen erreichbar. Über `permissions.read` steuern Sie den Zugriff auf den Endpunkt. Die eigentlichen Dienste konfigurieren Sie getrennt als `owsServices`, wobei `type` das Protokoll bestimmt (`wms`, `wfs`, `wmts`, `csw`).

%ref "gws.base.ows.server.action.Config"
