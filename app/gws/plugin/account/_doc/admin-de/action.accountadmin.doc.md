# Aktion "accountadmin" :/admin-de/konfiguration/action/accountadmin

Die Aktion `accountadmin` stellt die administrative Verwaltung von Benutzerkonten bereit. Über die Editier-Oberfläche legen Berechtigte Konten an, bearbeiten und löschen sie und setzen sie bei Bedarf zurück. Das Datenmodell der Konten kommt aus dem `adminModel` des [`account`-Helfers](/admin-de/konfiguration/helper/account); die Option `models` der Aktion wird nicht ausgewertet.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "accountadmin"
    permissions.read "allow admin, deny all"
}
```

`permissions.read` beschränkt die administrative Verwaltung auf berechtigte Nutzer (hier die Rolle `admin`). Welche Kontodaten in der Editier-Oberfläche bearbeitet werden und wer sie schreiben darf, legt das `adminModel` des `account`-Helfers fest (`isEditable`, `permissions.edit`); das Zurücksetzen eines Kontos verlangt Schreibrecht auf dieses Modell.

%ref "gws.plugin.account.admin_action.Config"
%demo "account_admin"
