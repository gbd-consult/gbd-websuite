# Aktion "select" :/admin-de/konfiguration/action/select

Die Aktion `select` aktiviert das Auswahl-Werkzeug im Client, mit dem Nutzer Objekte per Klick oder über Geometrien selektieren. Mit `tolerance` legen Sie die Klick-Toleranz fest, mit `storage` die Ablage für gespeicherte Auswahlen.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "select"
    tolerance "10px"
    storage {
        permissions {
            read "allow all"
            write "allow all"
            create "allow all"
        }
    }
}
```

`tolerance` legt fest, in welchem Umkreis um den Klickpunkt nach Objekten gesucht wird (Angabe mit Einheit, z. B. `"10px"`). `storage` aktiviert das Speichern der Auswahllisten mit eigenen Lese-, Schreib- und Anlege-Rechten.

%ref "gws.plugin.select_tool.action.Config"
%demo "select_tool"
