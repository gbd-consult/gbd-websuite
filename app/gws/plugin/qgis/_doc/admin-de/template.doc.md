# Vorlage "qgis" :/admin-de/konfiguration/template/qgis

Die Vorlage `qgis` verwendet ein Druck-Layout aus einem QGIS-Projekt als Vorlage. Die Karte wird von der WebSuite gerendert und mit der von QGIS erzeugten Ausgabe zu einem PDF kombiniert, sodass Gitter und andere Layout-Elemente über der Karte liegen. Das QGIS-Projekt geben Sie über `provider` an; das gewünschte Layout wählen Sie über `index` oder über den Titel der Vorlage.

## Das Drucklayout vorbereiten

Legen Sie das Layout in QGIS über den Layout-Manager an und geben Sie ihm einen sprechenden Namen – dieser Name erscheint im Client als Auswahl der Druckvorlage und dient zugleich der Zuordnung in der Konfiguration.

Entscheidend ist das Zusammenspiel beider Systeme: Die **Karte** rendert die WebSuite, das **Layout** liefert QGIS, und beide werden übereinandergelegt. Alles, was im Layout über der Karte liegen soll – Gitter, Rahmen, Nordpfeil, Maßstabsleiste, Titel, Legende –, gehört daher in das QGIS-Layout. Der Karteninhalt selbst kommt aus der Konfiguration der WebSuite und richtet sich danach, was der Nutzer im Client sieht.

%warn
Die Kartenelemente des Layouts müssen durchsichtig sein. Zeichnet QGIS an dieser Stelle einen deckenden Hintergrund, überdeckt er die von der WebSuite gerenderte Karte, und das PDF enthält nur das Layout. Setzen Sie Hintergrund und Füllung der Kartenelemente entsprechend transparent.
%end

Platzhalter für Seitenzahlen stehen zur Verfügung, sodass sich auch mehrseitige Layouts nummerieren lassen. Die Auflösung des Ergebnisses bestimmt nicht das Layout, sondern die [Qualitätsstufe des Druckers](/admin-de/konfiguration/printer).

## Beispiel-Konfiguration ::

```javascript
printers+ {
    template {
        type "qgis"
        provider.path "/data/qgis/print.qgs"
        index 0
    }
    qualityLevels [
        { dpi 72 name "Entwurf" }
        { dpi 150 name "Gute Qualität" }
    ]
}
```

Über `provider.path` verweisen Sie auf das QGIS-Projekt mit dem Drucklayout. `index` wählt das gewünschte Layout aus, wenn das Projekt mehrere enthält; `0` bezeichnet das erste. Die `qualityLevels` des Druckers legen die im Client wählbaren Auflösungen fest.

%ref "gws.plugin.qgis.template.Config"
%demo "print_fields_qgis"
%demo "qgis_dynamic_legend"
%demo "qgis_print"
%demo "qgis_print_html"
