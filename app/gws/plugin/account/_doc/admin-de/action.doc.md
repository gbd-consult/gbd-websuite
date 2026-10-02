# Aktion "account" :/admin-de/konfiguration/action/account

Die Aktion `account` stellt die Selbstbedienungs-Funktionen für Benutzerkonten bereit: das Onboarding, bei dem ein Nutzer über den per E-Mail zugesandten Freischaltungs-Code sein Passwort setzt und – falls konfiguriert – einen zweiten Faktor einrichtet. Es handelt sich nicht um eine offene Registrierung; die Konten werden zuvor administrativ angelegt (siehe [Benutzerkonten](/admin-de/themen/zugriff/konten)). Die zugehörige Client-Oberfläche ist der Dialog `Dialog.Account`.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "account"
    permissions.read "allow all"
}
```

Die Aktion selbst hat keine eigenen Optionen; sie stellt die Onboarding-Endpunkte bereit. `permissions.read "allow all"` gibt sie frei, damit auch noch nicht angemeldete Nutzer ihr Konto über den zugesandten Freischaltungs-Code aktivieren können. Die Kontenverwaltung, die Vorlagen für die Onboarding-E-Mail und die MFA-Regeln konfigurieren Sie getrennt im `account`-Helper (`helpers`).

%ref "gws.plugin.account.account_action.Config"
%demo "user_account"
