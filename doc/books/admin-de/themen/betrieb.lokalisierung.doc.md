# Lokalisierung :/admin-de/themen/betrieb/lokalisierung

Server und Client sind sprach- und ortsunabhängig. Global legen Sie die verfügbaren Sprachen und die Zeitzone fest; die erste Sprache gilt als Voreinstellung, und das Gebietsschema lässt sich pro Projekt anpassen. Sämtliche Kommunikation erfolgt in UTF-8.

Aus dem Gebietsschema ergeben sich die Formate für Datum, Uhrzeit und Zahlen, die in Vorlagen in verschiedenen Ausführlichkeiten zur Verfügung stehen.

## Sprachen festlegen

Die verfügbaren Gebietsschemata geben Sie unter `locales` an, als Liste von Kennungen der Form `de_DE`. Die erste Angabe ist die Voreinstellung:

```javascript
locales ["de_DE" "en_US"]
```

In der Applikation gesetzt gilt die Liste für alle Projekte; ein Projekt kann sie mit einer eigenen `locales`-Angabe vollständig ersetzen.

Ohne `locales` arbeitet die WebSuite mit `de_DE`. Das betrifft nicht nur die Oberfläche, sondern auch die Formatierung von Datum und Zahlen in Vorlagen und Ausgaben.

Eine angeforderte Sprache, die nicht in der Liste steht, führt **nicht** zu einem Fehler. Ein vollständiges Gebietsschema wie `fr_FR` ersetzt die WebSuite stillschweigend durch den ersten Eintrag. Eine zweibuchstabige Angabe wie `de` wird gegen die Liste aufgelöst; passt kein Eintrag, verwendet die WebSuite das übliche Gebietsschema der Sprache, auch wenn es nicht in der Liste steht (`fr` wird zu `fr_FR`).

## Mehrsprachige Projektadressen

Soll dasselbe Projekt unter mehreren Sprachen erreichbar sein, nehmen Sie die Sprache in die Adresse auf. Eine [Rewrite-Regel](/admin-de/konfiguration/web) gibt das Sprachkürzel als `localeUid` an die eingebaute Projektseite weiter:

```javascript
web.site {
    # z.B. http://site/project/meine-karte/de
    rewriteRules+ {
        pattern "^/project/([a-z0-9_.-]+)/([a-z][a-z])"
        target "/_/webPage/name/project/projectUid/$1/localeUid/$2"
    }
}
```

Die Adresse ohne Sprachkürzel, `/project/meine-karte`, beantwortet weiterhin die eingebaute Standardregel; das Projekt erscheint dann in der ersten Sprache aus `locales`. Hinter `/_/<Befehl>/` liest die WebSuite den Pfad paarweise als Parameter und Wert; so landen `projectUid` und `localeUid` als Angaben der Anfrage. Die eingebaute Projektseite gibt das ermittelte Gebietsschema selbst an den Client weiter.

Verwenden Sie stattdessen eine eigene Projektseite, geben Sie das ermittelte Gebietsschema dort selbst an den Client weiter. Es steht in der Vorlage als `locale` zur Verfügung:

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
