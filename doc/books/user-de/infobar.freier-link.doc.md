# Freier Verweis :/user-de/infobar.freier-link

Die GBD WebSuite bietet zwei Elemente zum Thema {title Freier Verweis} für die Infobar, um benutzerdefinierte Links zu integrieren. Diese Elemente ergänzen die vordefinierten Links wie `Infobar.About`, `Infobar.Help` und `Infobar.HomeLink` und ermöglichen die Integration zusätzlicher externer Verweise oder Dokumente.
                               
## Text-basiertes Element

Ein Textelement ```Infobar.Link```, das einen anklickbaren Hyperlink in der Infobar darstellt.

**Verwendungszweck:**
- Anzeige benutzerdefinierter Links als Textlinks in der unteren Informationsleiste
- Geeignet für ausgeschriebene Link-Beschriftungen
- Ideal für gut lesbare, beschreibende Links

**Eigenschaften:**
- `tag` (String, erforderlich): `"Infobar.Link"`
- `after` (String, optional): Positionierung nach einem bestimmten Element
- `before` (String, optional): Positionierung vor einem bestimmten Element
- `permissions` (PermissionsConfig, optional): Zugriffsberechtigungen für das Element

## Icon-basiertes Element

Ein Icon-basiertes Element ```Infobar.ButtonLink```, das einen anklickbaren Link als Button in der Infobar darstellt.

**Verwendungszweck:**
- Anzeige benutzerdefinierter Links als Icon-Buttons in der unteren Informationsleiste
- Platzsparende Alternative zu Textlinks
- Geeignet für standardisierte Icons (z.B. Social Media, externe Tools)

**Eigenschaften:**
- `tag` (String, erforderlich): `"Infobar.ButtonLink"`
- `after` (String, optional): Positionierung nach einem bestimmten Element
- `before` (String, optional): Positionierung vor einem bestimmten Element
- `permissions` (PermissionsConfig, optional): Zugriffsberechtigungen für das Element

**Unterschiede zwischen Text- und Icon-basiertem Element**

| Merkmal     | Infobar.Link        | Infobar.ButtonLink        |
|-------------|---------------------|---------------------------|
| Darstellung | Text                | Icon/Button               |
| Platzbedarf | Mittel bis hoch     | Gering                    |
| Lesbarkeit  | Selbsterklärend     | Benötigt erkennbares Icon |
| Verwendung  | Beschreibende Links | Standardisierte Verweise  |

**Hinweise**

- Die Konfiguration von URL, Link-Text oder Icon erfolgt über zusätzliche Eigenschaften, die in der erweiterten ElementConfig-Dokumentation beschrieben sind
- Beide Elemente sind Teil der Standard-UI-Komponenten und können beliebig mit anderen Infobar-Elementen kombiniert werden
- Die Reihenfolge der Elemente in der Liste bestimmt die Standardreihenfolge in der Infobar (sofern nicht durch after/before überschrieben)


