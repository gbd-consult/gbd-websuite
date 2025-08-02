# Referenz :/admin-de/reference

Nachfolgend werden die Konfigurationsobjekte der GBD Web Suite beschrieben. Unsere Konfiguration ist eine Stuktur, oder JSON-Objekt. Die Konfiguration ist in verschiedene Objekte unterteilt, die jeweils eine bestimmte Funktionalität oder einen bestimmten Bereich repräsentieren. Jedes Objekt enthält Eigenschaften, die konfiguriert werden können, um das Verhalten der Anwendung anzupassen. Als Eingeschaten eines Objekts können weitere Objekte verwendet werden, bzw. Listen von Objekten, "primitive" Werte (wie Ganzzahlen oder Strings), oder weitere atomaren Wertetypen, die eigene Semantik haben, wie z.B. `CrsName` oder `Duration`.

Jedes Element (Objekt oder Wertetyp) wird in dieser Referenz durch seinen Typ-Namen und ggf. seine Eigenschaften beschrieben. Die Typ-Namen (wie z.B. `gws.base.application.core.Config`) dienen nur Navigationszwecken und sind nicht Teil der Konfiguration selbst. Sie helfen jedoch, die Struktur und die Hierarchie der Konfiguration zu verstehen.

Das oberste Ebene der Konfiguration ist das [Hauptobjekt](/admin-de/reference/gws.base.application.core.Config). Es enthält alle anderen Objekte und deren Eigenschaften. 

Die unterschiedlichen Elemente sind mit diesen Tags gekennzeichnet:

- <span class="configref_category_object"></span> - ein Objekt, gefolgt von seinem Typ-Namen und Eigenschaften. Die Objekte werden in der JSON-Notation mit geschweiften Klammern `{}` dargestellt.

- <span class="configref_category_type"></span> - ein atomarer Wertetyp. Diese Werte werden in JSON als Strings bzw. Zahlen dargestellt, haben jedoch eine spezielle Bedeutung in der Konfiguration. Zum Beispiel `CrsName` für Koordinatenreferenzsysteme oder `Duration`{.configref_type} für Zeitdauern.

- <span class="configref_category_enum"></span> - eine vordefinierte Liste von Werten. An der entsprechenden Stelle in der Konfiguration kann dann ein Wert aus dieser Liste verwendet werden. Diese Werte werden in JSON als Strings dargestellt.

- <span class="configref_category_variant"></span> - ein Objekt, das eine Eigenschaft `type`{.configref_propname} enthält, die angibt, welcher Typ von Objekt hier verwendet wird. Diese Objekte können verschiedene Eigenschaften haben, abhängig von ihrem Typ.

Jedes Objekt kann nur Eigenschaften enthalten, die auf dieser Seite beschrieben sind. Die meisten Eigenschaften sind optional. Erforderliche Eigenschaften sind mit einem Sternchen (*) gekennzeichnet (z.B. `tag`{.configref_required}) und stehen immer oben in der Liste.

Die Eigenschaften können nur Typen annehmen, die hier dokumentiert sind. Folgende Bezeichungen werden verwendet, um die Typen zu kennzeichnen: 

| Schriftart                                           | Bezeichnung                                                                                                                                |
|------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [`gws.CrsName`{.configref_typename}](../gws.CrsName) | eine klickbare Referenz zu einem Objekt oder Wertetyp                                                                                      |
| `str`{.configref_typename}                           | ein String, wie `"test"`                                                                                                                   |
| `int`{.configref_typename}                           | eine Ganzzahl, wie `42`                                                                                                                    |
| `bool`{.configref_typename}                          | ein boolescher Wert, entweder `true` oder `false`                                                                                          |
| `float`{.configref_typename}                         | Reelzahl, wie `3.14`                                                                                                                       |
| ``dict``{.configref_typename}                        | generisches Schlüssel-Wert-Objekt, das in JSON als `{"schlüssel": wert...}` dargestellt wird                                               |
| TYP **[ ]**                                          | eine Liste vom Typ, z.B. **int [ ]** ist eine Liste von Ganzzahlen                                                                         |
| `wert`{.configref_literal}                           | ein literaler Wert, muss genau so geschrieben werden, wie er hier steht, z.B. `true`{.configref_literal} oder `"edit"`{.configref_literal} |



%include ../../../app/__build/configref.de.md
