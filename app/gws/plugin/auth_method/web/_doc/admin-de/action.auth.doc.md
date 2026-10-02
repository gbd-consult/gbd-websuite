# Aktion "auth" :/admin-de/konfiguration/action/auth

Die Aktion `auth` stellt die Anmelde-Schnittstelle für die webbasierte Authentifizierung bereit. Sie verarbeitet Login und Logout, die Abfrage des angemeldeten Nutzers sowie die Zwei-Faktor-Verifizierung. Sie setzt eine konfigurierte `web`-Authentifizierungsmethode voraus.

## Beispiel-Konfiguration ::

```javascript
auth.methods+ {
    type "web"
    secure true
    loginRedirect {
        pattern "^/(project)"
        target "/login"
    }
}

actions+ {
    type "auth"
    permissions.read "allow all"
}
```

Die Aktion selbst hat keine eigenen Optionen; sie stellt nur die Login-, Logout- und Prüf-Endpunkte bereit. Voraussetzung ist die `web`-Authentifizierungsmethode (`auth.methods`), die die Sitzung über ein Cookie führt. Mit `loginRedirect` leiten Sie nicht angemeldete Anfragen auf passende URLs zur Login-Seite um. `permissions.read "allow all"` gibt die Endpunkte frei, damit sich auch nicht angemeldete Nutzer anmelden können.

%ref "gws.plugin.auth_method.web.action.Config"
