# Referenz :/admin-de/reference

**Legende**

| Schriftart | Bezeichnung | Beispiel |
|---|---|---|
| `str`{.configref_typename} | Typ (String) | "test" |
| `int`{.configref_typename} | Typ (Integer) | 123 |
| `bool`{.configref_typename} | Typ (Bool) | true |
| `float`{.configref_typename} | Typ (Float) | 3.14 |
| **[** `str`{.configref_typename} **]** | eine List vom Typ | [ "test1" "test2" "test3" "test4"] |
| ``dict`` | generisches Schlüssel-Wert-Objekt | { "key1":"value1", "key2":"value2", "key3":"value3" }|
| **[** ``Typ`` **]** | Liste (Array) von Elementen vom ``Typ`` | [{"uid": "test1", type:"posgres"} {"uid": "test2", type:"posgres"}] |
| `access`{.configref_propname} | Eigenschaft | null (optional) |
| `access`{.configref_required} | erforderliche Eigenschaft | "test" (required) |
| `hello`{.configref_literal} | literaler Wert | "hello" |

**Konfiguration**

Das primäre Konfigurationsobjekt ist [](gws.base.application.Config).

**Konfigurationsobjekte**

%toc
/admin-de/reference/*
%end

%include ../../../app/__build/configref.de.html
