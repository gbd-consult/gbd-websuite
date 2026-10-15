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

Eine angeforderte Sprache, die nicht in der Liste steht, führt **nicht** zu einem Fehler. Ein vollständiges Gebietsschema wie `fr_FR` ersetzt die WebSuite stillschweigend durch den ersten Eintrag. Eine zweibuchstabige Angabe wie `de` wird gegen die Liste aufgelöst; passt kein Eintrag, verwendet die WebSuite das übliche Gebietsschema der Sprache, auch wenn es nicht in der Liste steht (`fr` wird zu `fr_FR`).

## Mehrsprachige Projektadressen

Soll dasselbe Projekt unter mehreren Sprachen erreichbar sein, nehmen Sie die Sprache in die Adresse auf. Zwei [Rewrite-Regeln](/admin-de/konfiguration/web) genügen — eine mit Sprachkürzel, eine ohne:

```javascript
web.site {
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
`withDefaultRewriteRules false` ist hier kein Beiwerk. Die eingebaute Standardregel `^/project/([a-z0-9_-]+)(?=/@|$)` wird sonst **vor** Ihre Regeln einsortiert und beantwortet die Adresse ohne Sprachkürzel selbst — mit der mitgelieferten Projektseite statt Ihrer Vorlage.
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

Die Zeitzone der WebSuite bestimmt die Umgebungsvariable `TZ` des Containers; das Image gibt `Europe/Berlin` vor. Eine abweichende Zone tragen Sie im `environment`-Block der `docker-compose.yml` ein, etwa `TZ=Europe/Vienna`. Die frühere Konfigurationsoption `server.timeZone` wird seit 8.5 ignoriert, und eine nach `/etc/localtime` eingehängte Zonendatei des Hosts wirkt nicht, solange `TZ` gesetzt ist.

%see
Siehe auch: [Konfiguration/Applikation](/admin-de/konfiguration/application), [Konfiguration/Projekt](/admin-de/konfiguration/project), [Konfiguration/Web](/admin-de/konfiguration/web).
%end
