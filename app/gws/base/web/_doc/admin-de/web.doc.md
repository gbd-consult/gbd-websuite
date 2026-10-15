# Web :/admin-de/konfiguration/web

Web-Inhalte liefert die GBD WebSuite über eine *Webseite* (Site) aus, die Sie unter `web.site` mit den Regeln für Auslieferung und Zugriff konfigurieren. Mehrere Webseiten werden nicht unterstützt; die Liste `web.sites` ist veraltet und darf höchstens einen Eintrag haben. `hostnames` nennt die Hostnamen, die die Webseite beantwortet; Anfragen an andere Hostnamen werden abgewiesen. Ohne `hostnames` beantwortet sie jeden Hostnamen. Die frühere Eigenschaft `host` ist veraltet.

Was zu den [Web-Inhalten](/admin-de/themen/grundlagen/andere-inhalte) zählt und wie sie sich von den Karteninhalten abgrenzen, ordnet das gleichnamige Thema ein.

## Verzeichnisse

Jede Webseite kennt zwei Verzeichnisse:

| Eigenschaft | Voreinstellung | Inhalt |
|---|---|---|
| `root` | `/data/web` | statische Dokumente, unverändert ausgeliefert |
| `assets` | `/data/assets`, sofern vorhanden | Dateien, die der Server vor der Auslieferung verarbeitet – Vorlagen und zugriffsbeschränkte Dateien |

Die Voreinstellungen greifen nur, wenn die Verzeichnisse existieren. Fehlt `/data/assets`, hat die Webseite **kein** Assets-Verzeichnis, und Anfragen darauf laufen ins Leere; fehlt `/data/web`, weicht der Server auf ein temporäres Verzeichnis aus und schreibt eine Warnung ins Protokoll.

Ein Projekt kann ein eigenes `assets`-Verzeichnis mitbringen; dessen Einstellungen gehen dann denen der Webseite vor.

## MIME-Filterung

Es wird **nicht** jede Datei ausgeliefert. Der Server prüft den MIME-Typ gegen eine eingebaute Liste erlaubter Typen; alles andere wird abgewiesen. Zwei Eigenschaften steuern das je Verzeichnis:

| Eigenschaft | Wirkung |
|---|---|
| `allowMime` | **ersetzt** die eingebaute Liste – nur die hier genannten Typen werden ausgeliefert |
| `denyMime` | schränkt die eingebaute Liste weiter ein |

```javascript
assets {
    dir "/data/assets"
    allowMime ["application/pdf" "image/png"]
}
```

%info
Lädt eine Datei nicht, obwohl sie vorhanden und der Pfad richtig ist, prüfen Sie zuerst ihren MIME-Typ. Ein nicht gelisteter Typ wird wortlos abgewiesen.
%end

## Rewrite-Regeln

Damit aus internen Aufrufen lesbare Adressen werden, kennt jede Webseite Rewrite-Regeln. Sie bilden eingehende URLs auf interne Aufrufe ab und bringen umgekehrt selbst erzeugte Adressen wieder in ihre lesbare Form. Das betrifft insbesondere den [dynamischen Endpunkt](/admin-de/themen/grundlagen/anfragen) und die Adressen der OWS-Dienste.

Sie führen die Regeln unter `rewriteRules` auf; jede Regel kennt drei Eigenschaften:

| Eigenschaft | Bedeutung |
|---|---|
| `pattern` | regulärer Ausdruck, gegen den die Adresse geprüft wird |
| `target` | Zieladresse, mit `$1`, `$2` … für die geklammerten Teile des Musters |
| `reversed` | Regel gilt für die Rückrichtung, Voreinstellung `false` |

Ein `target` ohne führenden Schrägstrich wird um einen ergänzt, sofern es keine vollständige URL ist.

Die Regeln der Hinrichtung wertet nicht die WebSuite aus: Sie landen als `rewrite`-Anweisungen in der Konfiguration des vorgeschalteten nginx, in der Reihenfolge der Liste. Die erste passende Regel gewinnt. Eine Änderung wirkt sich deshalb erst aus, wenn der Server die Konfiguration neu erzeugt hat.

### Die eingebauten Standardregeln

Jede Webseite erhält zwei Regeln, die auf die mitgelieferten Startseiten führen, und zwei Regeln für die Rückrichtung:

| Muster | Ziel | `reversed` |
|---|---|---|
| `^/$` | `/_/webPage/name/home` | |
| `^/project/([a-z0-9_-]+)(?=/@\|$)` | `/_/webPage/name/project/projectUid/$1` | |
| `^/_/webPage/name/home$` | `/` | `true` |
| `^/_/webPage/name/project/projectUid/([a-z0-9_-]+)` | `/project/$1` | `true` |

Diese Regeln werden **hinter** Ihre eigenen angehängt und greifen nur, wo keine Ihrer Regeln passt, in beiden Richtungen. Eine eigene Regel für `/project/…` geht der Standardregel also vor. Eine Standardregel, deren Muster zeichengleich bereits in Ihren Regeln steht, fügt die WebSuite nicht noch einmal hinzu.

%warn
Eine eigene Regel mit sehr allgemeinem Muster, etwa `^/(.*)`, fängt auch die Startseite und die Projektseiten ab, bevor die Standardregeln zum Zug kommen.
%end

Mit `withDefaultRewriteRules false` unterbleibt das Hinzufügen vollständig — der richtige Weg, wenn Sie eine Konfiguration aus einer älteren Version übernehmen, deren Adressen bereits vollständig von Hand gesetzt sind.

### Regeln für die Rückrichtung

Eine Regel mit `reversed true` beschreibt den umgekehrten Weg: Sie wird nicht an nginx weitergereicht, sondern von der WebSuite verwendet, wenn diese selbst eine öffentliche Adresse aus einem internen Pfad bildet. Nur so erscheinen in erzeugten Dokumenten — etwa in den Capabilities eines OWS-Dienstes — die lesbaren Adressen statt der internen.

### Veraltet: `rewrite`

Bis Version 8.4 hieß die Eigenschaft `rewrite`. Der alte Name wird weiterhin ausgewertet, aber **nur, wenn `rewriteRules` fehlt** — sind beide gesetzt, bleibt `rewrite` wirkungslos. Benennen Sie die Eigenschaft bei Gelegenheit um.

## Sicherheit und Zugriff

Auf Ebene der Webseite konfigurieren Sie außerdem:

- `cors` – Freigaben für Zugriffe aus anderen Herkünften,
- `contentSecurityPolicy` und `permissionsPolicy` – die entsprechenden HTTP-Header, beide mit einer restriktiven Voreinstellung,
- `errorPage` – eine eigene Vorlage für Fehlerseiten. Seit 8.4 veraltet; verwenden Sie stattdessen eine Vorlage mit dem Subject `application.error`.

Zertifikat (`crt`), Schlüssel (`key`) und die HSTS-Gültigkeitsdauer (`hsts`, Voreinstellung `365d`) gehören nicht zur Webseite, sondern stehen unter `web.ssl`.

## Beispiel-Konfiguration ::

```javascript
web.site {
    hostnames ["www.stadt.example"]
    root.dir "/data/web"
    assets.dir "/data/assets"
    rewriteRules+ {
        pattern "^/karte/([a-z0-9_-]+)$"
        target "/_/webPage/name/project/projectUid/$1"
    }
}
```

Diese Webseite beantwortet nur Anfragen an den Hostnamen `www.stadt.example`. `root.dir` benennt das Verzeichnis der unverändert ausgelieferten statischen Dokumente, `assets.dir` das der serverseitig verarbeiteten Dateien. Die Rewrite-Regel macht `/karte/basiskarte` zu einer zweiten Adresse für die Projektseite; `$1` übernimmt den im `pattern` geklammerten Teil.

Die eingebauten Standardregeln bleiben daneben aktiv, solange `withDefaultRewriteRules` nicht auf `false` gesetzt ist — das Projekt ist hier also sowohl unter `/karte/basiskarte` als auch unter `/project/basiskarte` erreichbar. Eine eigene Regel für `/project/…` wäre dagegen überflüssig, weil die Standardregel genau das bereits leistet.

%ref "gws.base.web.site.Config"
