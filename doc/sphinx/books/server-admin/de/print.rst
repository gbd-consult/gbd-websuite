Drucken
========

Ein Projekt in der GBD WebSuite kann mehrere Druckvorlagen bereitstellen. Für jede Vorlage können Sie einen DPI-Wert vergeben. Es ist außerdem möglich das unterschiedliche Qualitätsstufen ("Normal", "Hoch" und "Best") vorkonfigueriert werden. So muss dann kein DPI-Wert händisch mehr vergeben werden, sondern die gewünschte Qualität kann per Begriff gewählt werden. Beachten Sie, dass das Drucken mit hohen DPI-Werten viel Speicherplatz benötigt und nicht mit Quellen möglich ist, welche Beschränkungen für die Begrenzung von Anfragen auferlegen. Das Drucken einer A3-Karte mit 1200 DPI wird bei einer beschränkten Quelle nicht funktionieren.

Entwurfsdruck
--------------

Der GBD WebSuite Client unterstützt auch den Druckmodus "Draft" ("Screenshot"). Dieser führt nicht die eigentliche Kartendarstellung durch, sondern gibt die Bildschirmkarte als Bitmap-Bild aus.


Vorlagentypen
--------------

Html
~~~~

Eine html Druckvorlage ist eine ``mako`` Vorlage, die den ``{MAP}`` Platzhalter enthält, der beim Drucken durch das aktuelle Kartenbild ersetzt wird.

QGIS
~~~~

Eine QGIS-Vorlage ist eine Druckkomposition aus einer QGIS ``qgs`` Karte. Wenn es mehrere Kompositionen gibt, können Sie ``compositionIndex`` oder ``compositionName`` verwenden, um die zu verwendende Komposition zu identifizieren. Hinweis: Da Karten in der GBD WebSuite separat gerendert werden, müssen sowohl Vorlage als auch Kartenhintergrund in der Komposition auf ``None`` (transparent) gesetzt werden.
