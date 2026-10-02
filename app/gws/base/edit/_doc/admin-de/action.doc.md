# Aktion "edit" :/admin-de/konfiguration/action/edit

Die Aktion `edit` stellt das Backend für das Editieren von Vektorobjekten bereit. Sie muss aktiv sein, damit Nutzer im Client über die konfigurierten Editier-Modelle Objekte (Features) anlegen, ändern und löschen sowie verknüpfte Objekte bearbeiten können.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "edit"
}

models+ {
    type "postgres"
    tableName "edit.poi"
    title "Interessante Orte"
    isEditable true
    permissions.edit "allow all"

    fields+ {
        name "id"
        type "integer"
        isPrimaryKey true
        permissions.edit "deny all"
    }
    fields+ {
        name "name"
        type "text"
        title "Name"
        widget { type "input" }
    }
}
```

Die Aktion selbst hat keine eigenen Optionen; sie schaltet lediglich das Backend frei. Was editierbar ist, steuern die Modelle: `isEditable true` gibt ein Modell zum Bearbeiten frei, `permissions.edit` legt fest, wer schreiben darf. Auf Feldebene lässt sich das verfeinern – im Beispiel bleibt der Primärschlüssel `id` über `permissions.edit "deny all"` schreibgeschützt. Modelle können global (`models+`) oder direkt an einem Layer konfiguriert werden.

%ref "gws.base.edit.action.Config"
%demo "edit_auto"
%demo "edit_bbox"
%demo "edit_custom"
%demo "edit_lazy"
%demo "edit_multipolygon"
