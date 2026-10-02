# Helfer "email" :/admin-de/konfiguration/helper/email

Der Hilfsdienst `email` versendet E-Mails über einen SMTP-Server, etwa für Onboarding- oder Benachrichtigungs-Nachrichten. Den Server konfigurieren Sie unter `smtp` mit `host`, optional `login`/`password` und `timeout`; die Absenderadresse setzen Sie mit `mailFrom`.

Der Verbindungsmodus `mode` bestimmt die Verschlüsselung und ist **standardmäßig `ssl`** (implizite SSL-Verbindung). Daneben gibt es `tls` (unverschlüsselt beginnend, dann STARTTLS) und `plain` (ohne Verschlüsselung). Lassen Sie `port` auf `0`, wählt die WebSuite den zum Modus passenden Standard-Port: `plain` → 25, `ssl` → 465, `tls` → 587. Setzen Sie den Modus, ohne den Port anzupassen, ändert sich der verwendete Port also mit.

## Beispiel-Konfiguration ::

```javascript
helpers+ {
    type "email"
    mailFrom "noreply@stadt.example"
    smtp {
        host "smtp.stadt.example"
        login "gws-mailer"
        password "geheim"
    }
}
```

`smtp.host` benennt den SMTP-Server; `login` und `password` sind nur nötig, wenn dieser eine Authentifizierung verlangt. `mailFrom` ist die Absenderadresse, die verwendet wird, sofern eine einzelne Nachricht keine eigene angibt. `mode` und `port` sind hier nicht gesetzt, es gilt also der Standardmodus `ssl` mit Port 465.

%ref "gws.plugin.email_helper.Config"
