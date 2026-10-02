# Modell-Widget "password" :/admin-de/konfiguration/modelWidget/password

Das Widget `password` stellt ein Passwortfeld dar, in dem die Eingabe verdeckt angezeigt wird, und eignet sich für Textfelder mit sensiblen Inhalten. Mit `withShow` fügen Sie eine Schaltfläche hinzu, mit der sich das eingegebene Passwort einblenden lässt, mit `placeholder` legen Sie einen Platzhaltertext fest.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "passwort"
    type "text"
    title "Passwort"
    widget {
        type "password"
        withShow true
        placeholder "Passwort eingeben"
    }
}
```

Das `text`-Feld `passwort` wird als verdecktes Eingabefeld dargestellt. Mit `withShow true` fügen Sie eine Schaltfläche hinzu, mit der sich die Eingabe im Klartext einblenden lässt; `placeholder` bestimmt den Hinweistext im leeren Feld.

%ref "gws.plugin.model_widget.password.Config"
