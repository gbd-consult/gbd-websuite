# Aktion "map" :/admin-de/konfiguration/action/map

Die Aktion `map` liefert Kartenbilder aus und bearbeitet die kartenbezogenen Anfragen des Clients. Sie muss aktiv sein, damit die Karte eines Projekts im Client dargestellt werden kann.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "map"
}
```

Die Aktion hat keine eigenen Optionen; das Aktivieren genügt. Den Zugriff steuern Sie wie bei jeder Aktion über `access` bzw. `permissions`.

%ref "gws.base.map.action.Config"
