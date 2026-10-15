# Authentifizierungsmethode "web" :/admin-de/konfiguration/authMethod/web

Die Methode `web` ist die formularbasierte Anmeldung über die Weboberfläche. Nach erfolgreicher Anmeldung wird die Sitzung über ein Cookie geführt; einen zweiten Faktor bindet sie bei Bedarf ein. Über `cookieName`, `cookiePath` und `cookieSameSite` steuern Sie das Sitzungs-Cookie, mit `loginRedirect` leiten Sie abgewiesene Zugriffe auf eine Anmeldeseite um.

Wie jede Methode nimmt `web` Anmeldungen standardmäßig nur über HTTPS an (`secure`). `allowInsecureFrom` hat bei dieser Methode keine Wirkung; eine Anmeldung über unverschlüsseltes HTTP ist nur mit `secure false` möglich.

## Beispiel-Konfiguration ::

```javascript
auth.methods+ {
    type "web"
    cookieName "gws_auth"
    cookieSameSite "Lax"
    loginRedirect {
        pattern "^/projekt/"
        target "/login.html"
    }
}
```

Die Methode wird über `auth.methods+` aktiviert. `cookieName` und `cookieSameSite` bestimmen Namen und SameSite-Verhalten des Sitzungs-Cookies. `loginRedirect` leitet abgewiesene GET-Zugriffe (Status 401 oder 403), deren URL auf `pattern` passt, auf die unter `target` angegebene Anmeldeseite um. Ist gar keine Methode konfiguriert, wird `web` automatisch verwendet.

%ref "gws.plugin.auth_method.web.core.Config"
