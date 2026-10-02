# Multifaktor-Adapter "email" :/admin-de/konfiguration/authMultiFactorAdapter/email

Der Adapter `email` fügt als zweiten Faktor ein Einmalkennwort hinzu, das per E-Mail an den Nutzer versendet wird. Der Nutzer muss dazu eine hinterlegte E-Mail-Adresse besitzen, das Kennwort wird bei jedem Versuch neu erzeugt. Über `templates` passen Sie Betreff und Text der Versand-E-Mail an; der Versand nutzt den konfigurierten `email`-Helper.

## Beispiel-Konfiguration ::

```javascript
auth.mfa+ {
    uid "AUTH_MFA_EMAIL"
    type "email"
    templates+ {
        type "text"
        subject "email.subject"
        text "Ihr Anmeldecode"
    }
    templates+ {
        type "text"
        subject "email.body"
        text "Ihr Code lautet: {{otp}}"
    }
}
```

Der Adapter wird über `auth.mfa+` aktiviert; die `uid` referenziert der Nutzer über sein Attribut `mfaUid`. Die beiden `templates` liefern Betreff (`email.subject`) und Inhalt (`email.body`) der Versand-E-Mail; im Text steht der erzeugte Code über den Platzhalter `{{otp}}` zur Verfügung. Der Nutzer muss eine E-Mail-Adresse hinterlegt haben, und ein `email`-Helper muss für den Versand konfiguriert sein.

%ref "gws.plugin.auth_mfa.email.Config"
