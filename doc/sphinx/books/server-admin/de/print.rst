Drucken
========

Ein Projekt in der GBD WebSuite kann mehrere Druckvorlagen bereitstellen. Für jede Vorlage können Sie auch eine Liste der DPI für jede Qualitätsstufe ("Normal", "Hoch" und "Best") konfigurieren. Beachten Sie, dass das Drucken mit hohen DPI-Werten viel Speicherplatz benötigt und nicht einmal mit Quellen möglich ist, die Beschränkungen für die Begrenzung von Anfragen auferlegen. Das Drucken einer A3-Karte mit DPI 1200 wird wahrscheinlich nicht funktionieren.

Entwurfsdruck
--------------

Der GBD WebSuite Client unterstützt auch den Druckmodus "Draft" ("Screenshot"), der nicht die eigentliche Kartendarstellung durchführt, sondern die Bildschirmkarte als Bitmap-Bild ausgibt.


Vorlagentypen
--------------

Html
~~~~

Eine html Druckvorlage ist eine ``mako`` Vorlage, die den ``{MAP}`` Platzhalter enthält, der beim Drucken durch das aktuelle Kartenbild ersetzt wird.

QGIS
~~~~

Eine QGIS-Vorlage ist eine Druckkomposition aus einer QGIS ``qgs`` mape. Wenn es mehrere Kompositionen gibt, können Sie ``compositionIndex`` oder ``compositionName`` verwenden, um die zu verwendende Komposition zu identifizieren. Hinweis: Da Karten in GWS separat gerendert werden, müssen sowohl Vorlage als auch Kartenhintergrund in der Komposition auf None (transparent) gesetzt werden.

