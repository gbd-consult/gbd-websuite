# Lokalisierung :/admin-de/themen/betrieb/lokalisierung

Server und Client sind sprach- und ortsunabhängig. Global legen Sie die verfügbaren Sprachen und die Zeitzone fest; die erste Sprache gilt als Voreinstellung, und das Gebietsschema lässt sich pro Projekt anpassen. Sämtliche Kommunikation erfolgt in UTF-8.

Aus dem Gebietsschema ergeben sich die Formate für Datum, Uhrzeit und Zahlen, die in Vorlagen in verschiedenen Ausführlichkeiten zur Verfügung stehen.

## Sprachen festlegen

Die verfügbaren Gebietsschemata geben Sie unter `locales` an, als Liste von Kennungen der Form `de_DE`. Die erste Angabe ist die Voreinstellung:

```javascript
locales ["de_DE" "en_US"]
```

In der Applikation gesetzt gilt die Liste für alle Projekte; ein Projekt kann sie mit einer eigenen `locales`-Angabe vollständig ersetzen.

%warn
Ohne `locales` arbeitet die WebSuite mit `en_CA` — kanadischem Englisch. Das betrifft nicht nur die Oberfläche, sondern auch die Formatierung von Datum und Zahlen in Vorlagen und Ausgaben. Setzen Sie die Option deshalb auch dann, wenn Sie nur eine Sprache brauchen.
%end

Eine angeforderte Sprache, die nicht in der Liste steht, führt **nicht** zu einem Fehler: Die WebSuite weicht stillschweigend auf den ersten Eintrag aus. Eine zweibuchstabige Angabe wie `de` genügt dabei, sie wird gegen die Liste aufgelöst.

## Mehrsprachige Projektadressen

Soll dasselbe Projekt unter mehreren Sprachen erreichbar sein, nehmen Sie die Sprache in die Adresse auf. Zwei [Rewrite-Regeln](/admin-de/konfiguration/web) genügen — eine mit Sprachkürzel, eine ohne:

```javascript
web.sites+ {
    withDefaultRewriteRules false

    # z.B. http://site/project/meine-karte/de
    rewriteRules+ {
        pattern "^/project/([a-z0-9_.-]+)/([a-z][a-z])"
        target "/_/webAsset/projectUid/$1/path/project.cx.html/localeUid/$2"
    }
    # z.B. http://site/project/meine-karte
    rewriteRules+ {
        pattern "^/project/([a-z0-9_.-]+)"
        target "/_/webAsset/projectUid/$1/path/project.cx.html"
    }
}
```

Die Reihenfolge ist wesentlich: Die Regel mit Sprachkürzel muss zuerst stehen, sonst fängt die allgemeinere Regel die Adresse ab. Hinter `/_/<Befehl>/` liest die WebSuite den Pfad paarweise als Parameter und Wert; so landen `projectUid`, `path` und `localeUid` als Angaben der Anfrage.

%warn
`withDefaultRewriteRules false` ist hier kein Beiwerk. Die eingebaute Standardregel `^/project/([a-z0-9_-]+)$` wird sonst **vor** Ihre Regeln einsortiert und beantwortet die Adresse ohne Sprachkürzel selbst — mit der mitgelieferten Projektseite statt Ihrer Vorlage.
%end

In der Vorlage geben Sie das ermittelte Gebietsschema an den Client weiter. Es steht dort als `locale` zur Verfügung:

```html
<script id="gwsOptions" type="application/json">
    {
        "projectUid": "{project.uid}",
        "localeUid": "{locale.uid}"
    }
</script>
...
<script src="/_/webSystemAsset/localeUid/{locale.uid}/path/app.js"></script>
```

Beide Stellen sind nötig und haben verschiedene Aufgaben: `gwsOptions` teilt dem Client mit, in welchem Gebietsschema er arbeiten soll, während der Pfadteil vor `app.js` bestimmt, welches Sprachpaket ausgeliefert wird. Die Texte der Oberfläche stecken im Bundle, nicht in der Konfiguration — fehlt die Angabe bei `app.js`, erscheint die Oberfläche in der Standardsprache, obwohl Datums- und Zahlenformate schon stimmen.

## Zeitzone

Damit die WebSuite zuverlässig mit lokaler Zeit arbeitet, muss die Zeitzone auch im Container gesetzt sein. Dafür gibt es drei Wege: die Konfigurationsoption `server.timeZone` (Vorgabe `Europe/Berlin`), die Umgebungsvariable `TZ`, oder das Einhängen einer Zonendatei des Hosts nach `/etc/localtime`.

%see
Siehe auch: [Konfiguration/Applikation](/admin-de/konfiguration/application), [Konfiguration/Projekt](/admin-de/konfiguration/project), [Konfiguration/Web](/admin-de/konfiguration/web).
%end
