# Referenz :/admin-de/reference

@TODO some intro prose, add link to the top objekt: [](/admin-de/reference/gws.base.application.core.Config)

In dieser Referenz werden folgende Schriftarten und Bezeichnungeng verwendet. 

**Bezeichungen von Eigenschaften:**

| Schriftart                             | Bezeichnung                             |
|----------------------------------------|-----------------------------------------|
| `access`{.configref_propname}          | optionale Eigenschaft                   |
| `tag`{.configref_required}             | erforderliche Eigenschaft               |

**Bezeichungen von Typen:**

| Schriftart                             | Bezeichnung                             | Beispiel                                                                 |
|----------------------------------------|-----------------------------------------|--------------------------------------------------------------------------|
| `str`{.configref_typename}             | String                                  | `"test"`                                                                 |
| `int`{.configref_typename}             | Ganzzahl                                | `123`                                                                    |
| `bool`{.configref_typename}            | Bool                                    | `true`                                                                   |
| `float`{.configref_typename}           | Reelzahl                                | `3.14`                                                                   |
| ``dict``{.configref_typename}          | generisches Schlüssel-Wert-Objekt       | `{ "key1":"value1", "key2":"value2", "key3":"value3" }`                  |
| **[** `...`{.configref_typename} **]** | eine Liste vom Typ                      | `[ "test1" "test2" "test3" "test4" ]`                                    |
| `...`{.configref_literal}            | literaler Wert                          | `'hello'`                                                                |

**Objektarten:**

| Schriftart                             | Bezeichnung                                                           |
|----------------------------------------|-----------------------------------------------------------------------|
| `type`{.configref_category}            | ein primitiver Wert                                                   |
| `struct`{.configref_category}          | ein Objekt in Klammern `{...}`                                        |
| `enum`{.configref_category}            | ein Wert aus einer vordefinierter Liste                               |
| `variant`{.configref_category}         | ein Objekt aus einer vordefinierter Liste, mit dem Eigenschaft `type`{.configref_propname} |


%include ../../../app/__build/configref.de.md
