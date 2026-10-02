# Authentifizierungsmethode "token" :/admin-de/konfiguration/authMethod/token

Die Methode `token` erwartet ein Zugriffstoken in einem HTTP-Header und eignet sich für den maschinellen Zugriff über Schnittstellen. Den Namen des Headers geben Sie über `header` an. Mit `prefix` legen Sie ein vorangestelltes Schlüsselwort fest, etwa `Bearer`, sodass ein Header der Form `Authorization: Bearer <Token>` ausgewertet wird.

## Beispiel-Konfiguration ::

```javascript
auth.methods+ {
    type "token"
    header "X-Auth-Token"
    prefix "Bearer"
}
```

Die Methode wird über `auth.methods+` aktiviert. `header` gibt den Namen des HTTP-Headers an, in dem das Token übermittelt wird. Mit `prefix` erwarten Sie ein vorangestelltes Schlüsselwort, hier `Bearer`, sodass ein Header der Form `X-Auth-Token: Bearer <Token>` ausgewertet und der Token-Wert an die Provider weitergereicht wird. Ohne `prefix` wird der gesamte Header-Wert als Token verwendet.

%ref "gws.plugin.auth_method.token.Config"
