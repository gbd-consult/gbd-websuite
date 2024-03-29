# Datenablage :/admin-de/config/datenablage

Im GBD WebSuite Client besteht die Möglichkeit, bestimmte Objekte, wie Markierungen, Bemaßungen oder Auswahllisten abzuspeichern und später aufzurufen. Serverseitig wird dies mit der Funktion *Datenablage* (``storage``) unterstützt. Die Datenablage wird mit dem ``storage`` Helper (siehe [Helper](/admin-de/config/helper)) konfiguriert. Zusätzlich zu der Helper-Konfiguration muss auch die ``storage`` Aktion und die Client Elemente ``Storage.Read`` und ``Storage.Write`` aktiviert werden.

Die Ablage wird in *Kategorien* (``category``) unterteilt wobei jede Kategorie einer Client-Funktion entspricht. In jeder Kategorie kann unbegrenzte Anzahl von Einträgen gespeichert werden. Aktuell sind folgende Kategorien implementiert:

| OPTION | BEDEUTUNG |
|---|---|
| ``Alkis`` | Flurstückslisten, siehe [Alkis](/admin-de/plugin/alkis) |
| ``Annotate`` | vom Benutzer erstellte Markierungen |
| ``Dimension`` | Bemaßungen, siehe [FeatuBemaßungre](/admin-de/plugin/dimension) |
| ``Select`` | Auswahllisten |
| ``Styles`` | vom Benutzer editierte Style Eigenschaften |

## Helper ``storage``

%reference_de 'gws.base.storage.manager.Config'

In der Konfiguration des Helpers geben Sie an, welche User-Rollen den Zugriff zu bestimmten Ablagen-Kategorien  haben. Zu jeder Kategorie kann eine Liste von Regeln zugeordnet werden, die angeben welche Rollen die Einträge in dieser Kategorie erzeugen (``write``) oder lesen (``read``) kann, oder beides (``all``). Außerdem können Sie ein Sternchen (``*``) eingeben, das für alle Kategorien steht. Im folgenden Beispiel haben die Rollen ``nutzer`` und ``expert`` Lesezugriff auf alle Kategorien, und die Rolle ``expert`` Schreibzugriff auf ``Dimension``:

```javascript

"helpers": [
    ...
    {
        "type": "storage",
        "permissions": [
            {
                "category": "*",
                "mode": "read",
                "access": [
                    { "role": "nutzer", "type": "allow"},
                    { "role": "expert", "type": "allow"}
                ]
            },
            {
                "category": "Dimension",
                "mode": "write",
                "access": [
                    { "role": "expert", "type": "allow"}
                ]
            }
        ]
    }
    ...
]
```