#!/usr/bin/env bash

DB=crs.sqlite

rm -f $DB
rm -f spatial_ref_sys.sql

curl -O 'https://raw.githubusercontent.com/postgis/postgis/svn-trunk/spatial_ref_sys.sql'

cat << EOF | sqlite3 $DB
CREATE TABLE spatial_ref_sys
(
    srid INTEGER PRIMARY KEY,
    auth_name TEXT,
    auth_srid INTEGER,
    srtext    TEXT,
    proj4text TEXT,
    units TEXT,
    longlat INTEGER
)  WITHOUT ROWID;

CREATE TABLE crs
(
    srid INTEGER PRIMARY KEY,
    proj4text TEXT,
    units TEXT,
    longlat INTEGER
)  WITHOUT ROWID;


EOF

cat spatial_ref_sys.sql | sqlite3 $DB

rm -f spatial_ref_sys.sql

cat << EOF | sqlite3 $DB
    INSERT INTO crs SELECT srid,proj4text,NULL,0 FROM spatial_ref_sys;
    DROP TABLE spatial_ref_sys;

    UPDATE crs SET units='ft'   WHERE proj4text LIKE '%units=ft%';
    UPDATE crs SET units='link' WHERE proj4text LIKE '%units=link%';
    UPDATE crs SET units='m'    WHERE proj4text LIKE '%units=m%';
    UPDATE crs SET units='us'   WHERE proj4text LIKE '%units=us%';

    UPDATE crs SET longlat=1 WHERE proj4text LIKE '%proj=longlat%';

    VACUUM;
EOF

echo 'done'
