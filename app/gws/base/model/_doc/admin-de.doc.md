# Datenmodelle :/admin-de/config/model





    %dbgraph 'Tabellen "house" und "street"'

        house (
            id integer pk,
            address varchar,
            floors integer,
            geom geometry,
            street_id integer fk
        )

        street (
            id integer pk,
            name varchar,
            geom geometry
        )

        house.street_id -> street.id

    %end

Eine Variante der "generic association" ohne Tabellen ID, d.h. das Feld kann mit beliebigen Tabellen verknüpft werden. In der Konfiguration steht lediglich der "Fremdschlüssel", die `relation` lässt man weg.


    %dbgraph 'Beispiel: Ein Baum, eine Bushaltestelle oder eine Laterne können dazugehörige Bilder haben.'

        tree (id integer pk, ...)
        busStop  (id integer pk, ...)
        lampPost  (id integer pk, ...)

        image (
            id integer pk,
            ...,
            table_id varchar,
            object_id integer,
        )

        tree.id ->> image.object_id
        busStop.id ->> image.object_id
        lampPost.id ->> image.object_id

    %end

Eine Variante der "generic association" ohne Tabellen ID, d.h. das Feld kann mit beliebigen Tabellen verknüpft werden. In der Konfiguration steht lediglich der "Fremdschlüssel", die `relation` lässt man weg.
