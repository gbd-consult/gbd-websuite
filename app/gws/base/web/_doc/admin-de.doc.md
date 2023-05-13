# Web Server :/admin-de/config/web

%reference_de 'gws.base.application.WebConfig'

Web-Inhalte (HTML-Dateien, Bilder, Downloads) werden in der GBD WebSuite von dem integrierten NGINX, einem schnellen und zuverlässigen Web-Server, verarbeitet.

Websiten
--------

Sie können mehrere *Webseiten* (`site`) in einer einzigen GBD WebSuite Installation konfigurieren, jeder mit seinem eigenen Hostnamen und Dokumentenstamm. Mit dem Hostnamen `*` (Sternchen) wird eine Default-Site bezeichnet.

Dokument-Ordner
---------------

Pro Webseite kann ein `root` Ordner konfiguriert werden, wo Sie Ihre statische Inhalte, wie Javascript, CSS Daten und Bilder, platzieren, sowie einen `assets` Ordner, der dynamische Inhalte, wie Vorlagen, enthält.

Per Default, werden aus diesen Ordnern nur folgende Dateierweiterungen zurückgegeben:

    css, csv, gif, html, jpeg, jpg, js, json, pdf, png, svg, ttf, txt, xml, zip, gml2, gml3

Sie können Ihre eigene Liste mit `allowMime` konfigurieren bzw. bestimmte Typen mit `denyMime` ausschließen.

Asset-Ordner können auf Projekt-Basis umkonfiguriert werden, zusätzlich muss die `asset` Aktion global oder projektweise freigeschaltet werden.  Bei Anfragen ohne Projekt-ID, wie z.B.

    http://example.com/_?cmd=assetHttpGetPath&path=mypage.html

wird die Vorlage `mypage.html` nur in Website Asset-Ordner gesucht, bei Anfragen mit Projekt-ID

    http://example.com/_?cmd=assetHttpGetPath&path=mypage.html&projectUid=myproject

dann wird das Asset zuerst in den Projekt-Assets gesucht, wenn es nicht gefunden wird, wird das Site-Asset-Verzeichnis als Fallback verwendet.

Rewrite-Regeln
--------------

*Rewrite-Regel* ermöglicht es Ihnen, komplizierte GET-Anfragen in einer einfacheren Form darzustellen. Eine Regel besteht aus zwei Komponenten: das `match` ist ein regulärer Ausdruck und das `target` ist die endgültige URL mit Platzhaltern `$1`, `$2` usw , die den Capture-Gruppen `(...)` im diesem Ausdruck entsprechen. Wenn die Ziel-URL absolut ist (beginnt mit einem Schema), führt der Server einen Redirect statt eines Rewritings durch.

Zum Beispiel, diese "schöne" URLs

    http://example.com/pages/products
    http://example.com/pages/services

können mit dieser Regel

    {
        "match": "^/pages/([a-z]+)",
        "target": "_?cmd=assetHttpGetPath&path=page_$1.html"
    }

in folgende Asset-Anfragen umgewandelt werden

    http://example.com/_?cmd=assetHttpGetPath&path=page_products.html
    http://example.com/_?cmd=assetHttpGetPath&path=page_services.html

Für die URLs, die vom Server selbst erzeugt werden, können die umkehrende Regel (`reversedRewrite`) definiert werden, die complexe URLs in die "schönen" umwandeln.

Fehlerseitenvorlage
-------------------

Sie können eine Fehlerseitenvorlage  (`errorPage`) konfigurieren, die bei HTTP Fehlern, wie `404 Not Found` gezeigt werden. Diese Vorlage hat Zugriff auf den Fehlercode in der Variablen `error`. Sie können den Befehl `@if` verwenden, um je nach Fehler unterschiedliche Inhalte bereitzustellen:

    <h1>Error!</h1>

    @if error == 404
        Datei nicht gefunden!
    @elif error == 403
        Zugriff verweigert!
    @else
        Sonstige Fehler {error}
    @end

SSL Konfiguration
-----------------

SSL kann unter `web.ssl` konfiguriert werden. Sie müssen Pfade (wie im Container sichtbar) zu Ihrem Zertifikatspaket (*bundle*) und dem privaten Schlüssel angeben. Die SSL-Konfiguration wird auf den gesamten Server angewendet, nicht nur auf einzeln konfigurierten Seiten.

    "web": {
        ...
        "ssl": {
            "crt": "/path/to/your-certificate-bundle.crt",
            "key": "/path/to/your-private-key.crt"
        }
    }

Wenn Ihr Zertifikat aus separaten Dateien besteht (z. B. Domainzertifikat, Zwischenzertifikat und Stammzertifikat), können Sie das Bundle erstellen, indem Sie sie zusammenfügen

    cat domain.crt intermediate.crt root.crt > bundle.crt
