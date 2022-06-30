## Relationsfelder

Diese Felder beschreiben Verbindungen zwischen Modellen. Grundsätzlich müssen beide Seiten der Verbindung konfiguriert werden. Besteht eine Verbindung zwischen den Modellen A und B, müssen in A und B Felder vorhanden sein, die miteinander über eine `relation` verknüpft sind:

```
// im Model A
{
    name "relation_zu_b"
    type "..."
    relation { modelUid "B" field "relation_zu_a" }
}

// im Model B
{
    name "relation_zu_a"
    type "..."
    relation { modelUid "A" field "relation_zu_b" }
}

```

Relationsfelder sind "virtual", d.h. sie repräsentieren keine aktuellen Spalten. Die Spalten, in denen die Verbindung
gespeichert ist (z.B. Fremdschlüssel), müssen in der Konfiguration explizit genannt werden.

Der Werte eines Relationsfeldes sind entweder ein Feature (Datensatz) bei `M:1` Verbindungen, oder eine Liste von Features bei `1:M` bzw `M:N` Verbindungen.

Es werden diverse Typen von Relationen unterstützt.

### `relatedFeature`

@db_diagram
Beispiel: Ein Haus gehört zu einer Strasse.
@db_table "house" "PK id|...|FK street_id"
@db_table "street" "PK id|..."
@db_arrow_1 "house:street_id" "street:id"
@end

Der Relationsfeld type `relatedFeature` beschreibt eine `M:1` oder "child-parent" Verbindung. In der Konfiguration muss das "parent" Modell, sowie der Fremdschlüssel angegeben werden.

```
// im Modell "house"

{
    name "street"
    type  "relatedFeature"
    title "Strasse"
    relation {
        modelUid "street"
        fieldName "houses"
    }
    foreignKey {
        name "street_id"
    }
}
```

### `relatedFeatureList`

@db_diagram
Beispiel: Eine Strasse hat mehrere Häuser.
@db_table "street" "PK id|..."
@db_table "house"  "PK id|...|FK street_id"
@db_arrow_m "street:id" "house:street_id"
@end

Ein Gegensatz zum type `relatedFeature`, beschreibt der type `relatedFeatureList` eine `1:M` Verbindung.

```
// im Modell "street"

{
    name "houses"
    type  "relatedFeatureList"
    title "Häuser"
    relation {
        modelUid "house"
        fieldName "street"
    }
}
```

### `relatedMultiFeatureList`

@db_diagram
Beispiel: Eine Strasse hat mehrere Objekte wie Bäume, Laternen und Bushaltestellen.
@db_table "street"   "PK id|..."
@db_table "tree"     "PK id|...|FK street_id"
@db_table "busStop"  "PK id|...|FK street_id"
@db_table "lampPost" "PK id|...|FK street_id"
@db_arrow_m "street:id" "tree:street_id"
@db_arrow_m "street:id" "busStop:street_id"
@db_arrow_m "street:id" "lampPost:street_id"
@end

Dieser type `relatedMultiFeatureList` beschreibt eine `1:M` Verbindung zwischen mehreren Modellen. In der Konfiguration muss mit `relations` alle verknüpften Felder mit Namen aufgelistet werden.

```
// im Modell "street"

{
    name "objects"
    type  "relatedFeatureList"
    title "Objekte"
    relations [
        { title "Baum"            modelUid "tree"     fieldName "street" }
        { title "Bushaltestelle"  modelUid "busStop"  fieldName "street" }
        { title "Laterne"         modelUid "lampPost" fieldName "street" }
    }
}
```

### `relatedLinkedFeatureList`

@db_diagram
Beispiel: Eine Strasse kann zu mehreren Stadtteilen gehören, zu einem Stadtteil gehören mehrere Strassen.
@db_table "street"   "PK id|..."
@db_table "district" "PK id|..."
@db_table "link" "FK street_id|FK district_id"
@db_arrow_m "street:id"   "link:street_id"
@db_arrow_m "district:id" "link:district_id"
@end

Beschreibt eine `M:N` Relation über eine Link-Tabelle. An beiden Seiten der Relation muss die Link-Tabelle sowie der zu dieser Seite gehörende Link-Schlüssel definiert werden.

```
// im Modell "street"

{
    name "districts"
    type  "relatedLinkedFeatureList"
    title "Stadtteile"
    relation {
        modelUid "district"
        fieldName "streets"
    }
    link {
        tableName "link"
        keyName "street_id"
    }
}

// im Modell "district"

{
    name "streets"
    type  "relatedLinkedFeatureList"
    title "Strassen"
    relation {
        modelUid "street"
        fieldName "districts"
    }
    link {
        tableName "link"
        keyName "district_id"
    }
}
```

### `relatedDiscriminatedFeature`

@db_diagram
Beispiel: Ein Bild ist mit einem Baum, einer Bushaltestelle oder Laterne verknüpft.
@db_table "tree"     "PK id|..."
@db_table "busStop"  "PK id|..."
@db_table "lampPost" "PK id|..."
@db_table "image"    "PK id|...|table_id|object_id"
@db_arrow_1 "image:object_id" "tree:id"
@db_arrow_1 "image:object_id" "busStop:id"
@db_arrow_1 "image:object_id" "lampPost:id"
@end

Beschreibt eine
so-genannte ["generic association"](https://docs.sqlalchemy.org/en/14/_modules/examples/generic_associations/generic_fk.html)
, wobei ein virtuller "Fremdschlüssel" gleichzeitig auf mehrere Tabellen verweist. Neben dem "Fremdschlüssel" wird auch eine Tabellen-ID ("discriminator") gespeichert. In der Konfiguration müssen beide "Schlüssel" und alle Relationen mit dazugehörigen Discriminator-Werten angegeben werden.

```
// im Modell "image"

{
    name "object"
    type  "relatedDiscriminatedFeature"
    title "Objekt"
    foreignKey {
        name "object_id"
    }
    discriminatorKey {
        name "table_id"
    }
    relations [
        { title "Baum"            modelUid "tree"     discriminator "T"  fieldName "images" }
        { title "Bushaltestelle"  modelUid "busStop"  discriminator "B"  fieldName "images" }
        { title "Laterne"         modelUid "lampPost" discriminator "L"  fieldName "images" }
    ]
}
```

### `relatedDiscriminatedFeatureList`

@db_diagram
Beispiel: Ein Baum, eine Bushaltestelle oder eine Laterne können dazugehörige Bilder haben.
@db_table "tree"     "PK id|..."
@db_table "busStop"  "PK id|..."
@db_table "lampPost" "PK id|..."
@db_table "image"    "PK id|...|table_id|object_id"
@db_arrow_m "tree:id"     "image:object_id"
@db_arrow_m "busStop:id"  "image:object_id"
@db_arrow_m "lampPost:id" "image:object_id"
@end

Das Gegenstück zu `relatedDiscriminatedFeature`. In der Konfiguration muss nur die Relation angegeben werden.

```
// in den Modellen "tree", "busStop" und "lampPost"

{
    name "images"
    type  "relatedDiscriminatedFeatureList"
    title "Bilder"
    relation {
        modelUid "image"
        fieldName "object"
    }
}
```

### `relatedGenericFeature`

@db_diagram
Beispiel: Ein Bild kann mit einem beliebigen Objekt verknüpft werden.
@db_table "...A" "PK id|..."
@db_table "...B" "PK id|..."
@db_table "...C" "PK id|..."
@db_table "image" "PK id|...|object_id"
@db_arrow_1 "image:object_id" "A:id"
@db_arrow_1 "image:object_id" "B:id"
@db_arrow_1 "image:object_id" "C:id"
@end

Eine Variante der "generic association" ohne Tabellen ID, d.h. das Feld kann mit beliebigen Tabellen verknüpft werden. In der Konfiguration steht lediglich der "Fremdschlüssel", die `relation` lässt man weg.

```
// im Modell "image"

{
    name "object"
    type  "relatedGenericFeature"
    title "Objekt"
    foreignKey {
        name "object_id"
    }
}
```

### `relatedGenericFeatureList`

@db_diagram
Beispiel: Beliebige Objekte können dazugehörigen Bilder haben.
@db_table "...A" "PK id|..."
@db_table "...B" "PK id|..."
@db_table "...C" "PK id|..."
@db_table "image" "PK id|...|object_id"
@db_arrow_m "A:id"     "image:object_id"
@db_arrow_m "B:id"  "image:object_id"
@db_arrow_m "C:id" "image:object_id"
@end

Das Gegenstück zu `relatedGenericFeature`. In der Konfiguration muss nur die Relation angegeben werden.

```
// in einem beliebigen Modell

{
    name "images"
    type  "relatedGenericFeatureList"
    title "Bilder"
    relation {
        modelUid "image"
        fieldName "object"
    }
}
```

