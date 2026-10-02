# Layer "group" :/admin-de/konfiguration/layer/group

Ein `group`-Layer fasst mehrere Layer zu einem gemeinsamen Baumknoten zusammen. Die enthaltenen Layer geben Sie über `layers` an; die Gruppe übernimmt deren räumliche Ausdehnung, Auflösungen und Legenden und lässt sich als Einheit ein- und ausblenden. Verwenden Sie diesen Typ, um verwandte Layer im Client übersichtlich zu strukturieren.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "Kultur"
    type "group"
    clientOptions.expanded true

    layers+ {
        title "Buchladen"
        type "geojson"
        provider.path "/demos/poi/poi.buchladen.geojson"
    }
    layers+ {
        title "Museum"
        type "geojson"
        provider.path "/demos/poi/poi.museum.geojson"
    }
}
```

`layers` enthält die zusammengefassten Layer; die Gruppe übernimmt deren Ausdehnung und Auflösungen und lässt sich im Client gemeinsam ein- und ausblenden. `clientOptions.expanded` öffnet den Gruppenknoten beim Laden. Für Hintergrundkarten setzen Sie stattdessen `clientOptions.exclusive true`, sodass im Client jeweils nur ein enthaltener Layer sichtbar ist.

%ref "gws.base.layer.group.Config"
%demo "client_options"
