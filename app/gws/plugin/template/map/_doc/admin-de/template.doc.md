# Vorlage "map" :/admin-de/konfiguration/template/map

Die Vorlage `map` gibt ausschließlich die Karte aus, ohne umgebendes HTML-Gerüst. Sie eignet sich, wenn Sie allein das Kartenbild als HTML, PDF oder PNG benötigen. Die Größe der Ausgabe richtet sich nach der Seitengröße der Vorlage.

## Beispiel-Konfiguration ::

```javascript
printers+ {
    template {
        type "map"
        pageSize [ "200mm" "200mm" ]
    }
}
```

Als Druckvorlage erzeugt der Typ `map` ein reines Kartenbild ohne umgebendes Layout. `pageSize` bestimmt die Abmessungen der Ausgabe; das gerenderte Kartenbild füllt diese Fläche vollständig aus.

%ref "gws.plugin.template.map.Config"
