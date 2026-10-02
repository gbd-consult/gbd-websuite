# Aktion "accountadmin" :/admin-de/konfiguration/action/accountadmin

Die Aktion `accountadmin` stellt die administrative Verwaltung von Benutzerkonten bereit. Über die Editier-Oberfläche legen Berechtigte Konten an, bearbeiten und löschen sie und setzen sie bei Bedarf zurück. Mit `models` konfigurieren Sie die Datenmodelle der Konten.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "accountadmin"
    permissions.read "allow admin, deny all"
    models+ {
        type "postgres"
        tableName "edit.nutzer"
        title "Benutzer"
        isEditable true
        permissions.edit "allow admin, deny all"

        fields+ {
            name "id"
            type "integer"
            isPrimaryKey true
            permissions.edit "deny all"
        }
        fields+ { name "login" type "text" title "Anmeldename" }
        fields+ { name "email" type "text" title "E-Mail" }
    }
}
```

`permissions.read` beschränkt die administrative Verwaltung auf berechtigte Nutzer (hier die Rolle `admin`). `models` legt fest, welche Kontodaten in der Editier-Oberfläche bearbeitet werden: `isEditable` gibt das Modell frei, `permissions.edit` steuert das Schreibrecht, und auf Feldebene lässt sich der Primärschlüssel `id` über `permissions.edit "deny all"` schreibschützen.

%ref "gws.plugin.account.admin_action.Config"
%demo "account_admin"
