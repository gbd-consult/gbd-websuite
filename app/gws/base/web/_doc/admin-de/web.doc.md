# Web :/admin-de/konfiguration/web

Web-Inhalte liefert die GBD WebSuite über *Webseiten* (Sites) aus – jeweils ein Hostname mit den zugehörigen Regeln für Auslieferung und Zugriff. Die Webseite mit dem Hostnamen `*` gilt, wenn keine andere passt.

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

Sie führen die Regeln unter `rewriteRules` auf; jede Regel kennt vier Eigenschaften:

| Eigenschaft | Bedeutung |
|---|---|
| `pattern` | regulärer Ausdruck, gegen den die Adresse geprüft wird |
| `target` | Zieladresse, mit `$1`, `$2` … für die geklammerten Teile des Musters |
| `reversed` | Regel gilt für die Rückrichtung, Voreinstellung `false` |
| `options` | zusätzliche Angaben für einzelne Regeltypen |

Ein `target` ohne führenden Schrägstrich wird um einen ergänzt, sofern es keine vollständige URL ist.

Die Regeln der Hinrichtung wertet nicht die WebSuite aus: Sie landen als `rewrite`-Anweisungen in der Konfiguration des vorgeschalteten nginx, in der Reihenfolge der Liste. Die erste passende Regel gewinnt. Eine Änderung wirkt sich deshalb erst aus, wenn der Server die Konfiguration neu erzeugt hat.

### Die eingebauten Standardregeln

Jede Webseite erhält zwei Regeln, die auf die mitgelieferten Startseiten führen:

| Muster | Ziel |
|---|---|
| `^/$` | `/_/webPage/name/home` |
| `^/project/([a-z0-9_-]+)$` | `/_/webPage/name/project/projectUid/$1` |

Diese Regeln werden **vor** Ihre eigenen einsortiert und gewinnen damit gegen sie. Ob eine Standardregel bereits vorhanden ist, prüft die WebSuite über einen **zeichengenauen Vergleich des Musters** — nicht darüber, ob Ihre Regel dieselben Adressen abdeckt.

%warn
Ein Muster, das dasselbe leistet, aber anders geschrieben ist, verhindert die Standardregel **nicht**. `^/project/([a-z0-9_.-]+)` unterscheidet sich als Zeichenkette von `^/project/([a-z0-9_-]+)$`; die Standardregel wird zusätzlich eingefügt, steht davor und beantwortet die Anfrage. Das Symptom ist eine Projektseite, die unerwartet die eingebaute Vorlage zeigt statt Ihrer eigenen. Entweder übernehmen Sie das Muster zeichengleich, oder Sie schalten die Standardregeln ab.
%end

Mit `withDefaultRewriteRules false` unterbleibt das Hinzufügen vollständig — der richtige Weg, wenn Sie eine Konfiguration aus einer älteren Version übernehmen, deren Adressen bereits vollständig von Hand gesetzt sind.

### Regeln für die Rückrichtung

Eine Regel mit `reversed true` beschreibt den umgekehrten Weg: Sie wird nicht an nginx weitergereicht, sondern von der WebSuite verwendet, wenn diese selbst eine öffentliche Adresse aus einem internen Pfad bildet. Nur so erscheinen in erzeugten Dokumenten — etwa in den Capabilities eines OWS-Dienstes — die lesbaren Adressen statt der internen.

### Veraltet: `rewrite`

Bis Version 8.4 hieß die Eigenschaft `rewrite`. Der alte Name wird weiterhin ausgewertet, aber **nur, wenn `rewriteRules` fehlt** — sind beide gesetzt, bleibt `rewrite` wirkungslos. Benennen Sie die Eigenschaft bei Gelegenheit um.

## Sicherheit und Zugriff

Auf Ebene der Webseite konfigurieren Sie außerdem:

- `ssl` – Zertifikat (`crt`), Schlüssel (`key`) und die HSTS-Gültigkeitsdauer (`hsts`, Voreinstellung `365d`),
- `cors` – Freigaben für Zugriffe aus anderen Herkünften,
- `contentSecurityPolicy` und `permissionsPolicy` – die entsprechenden HTTP-Header, beide mit einer restriktiven Voreinstellung,
- `errorPage` – eine eigene Vorlage für Fehlerseiten. Seit 8.4 veraltet; verwenden Sie stattdessen eine Vorlage mit dem Subject `application.error`.

## Beispiel-Konfiguration ::

```javascript
web.sites+ {
    host "www.stadt.example"
    root.dir "/data/web"
    assets.dir "/data/assets"
    rewriteRules+ {
        pattern "^/karte/([a-z0-9_-]+)$"
        target "/_/webPage/name/project/projectUid/$1"
    }
}
```

Diese Webseite gilt für den Hostnamen `www.stadt.example`. `root.dir` benennt das Verzeichnis der unverändert ausgelieferten statischen Dokumente, `assets.dir` das der serverseitig verarbeiteten Dateien. Die Rewrite-Regel macht `/karte/basiskarte` zu einer zweiten Adresse für die Projektseite; `$1` übernimmt den im `pattern` geklammerten Teil.

Die eingebauten Standardregeln bleiben daneben aktiv, solange `withDefaultRewriteRules` nicht auf `false` gesetzt ist — das Projekt ist hier also sowohl unter `/karte/basiskarte` als auch unter `/project/basiskarte` erreichbar. Eine eigene Regel für `/project/…` wäre dagegen überflüssig, weil die Standardregel genau das bereits leistet.

%ref "gws.base.web.site.Config"
