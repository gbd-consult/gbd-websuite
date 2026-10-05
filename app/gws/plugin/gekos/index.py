"""GekoS index, loaded from gek-online."""


import math

import gws
import gws.lib.shape
import gws.config.util
import gws.lib.crs
import gws.lib.xmlx
import gws.lib.net
import gws.lib.sa as sa
from . import core


"""
Gekos-Online can be called with different "instance" parameters or no "instance" at all

The  xml structure is like this:
    
    <?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
    <OnlineTreffer>
      <Vorgang>
        <AntragsartID>..</AntragsartID>
        <SystemNr>...</SystemNr>
        <X>407000.000</X>
        <Y>5716000.000</Y>
        <ObjectID>1</ObjectID>
        <Verfahren>...</Verfahren>
        <AntragsartBez>...</AntragsartBez>
        <Darstellung>...</Darstellung>
        <Massnahme>....</Massnahme>
        <Tooltip>...</Tooltip>
        <UrlFV>...</UrlFV>
        <UrlOL>...</UrlOL>
      </Vorgang>
      <Vorgang>
    ....
    
ObjectID appears to be unique within an instance, so we generate a PK = instance_ObjectID  
    
"""


class Object(gws.Node):
    """GekoS index.

    Loads GekoS records from gek-online and stores them in a PostGIS table.
    """

    db: gws.DatabaseProvider
    """Database provider for the index table."""
    tableName: str
    """Name of the index table."""
    position: core.PositionConfig
    """Position correction for points, or ``None``."""
    crs: gws.Crs
    """CRS of the GekoS coordinates."""

    def configure(self):
        gws.config.util.configure_database_provider_for(self, ext_type='postgres')
        self.tableName = self.cfg('tableName')
        self.crs = gws.lib.crs.get(self.cfg('crs'))
        self.position = self.cfg('position')

    def create(self):
        """Load the records of all sources and recreate the index table.

        The table is dropped, created again and filled with the records.
        """
        recs = self._collect()
        self._write(recs)

    def _collect(self):
        """Load and transform the records of all sources."""
        recs = []

        for source in self.cfg('sources'):
            rs = self._load(source)
            gws.log.info(f'loaded {len(rs)} records from {source.instance!r}')
            rs = self._transform(rs, source.instance)
            recs.extend(rs)

        return recs

    def _load(self, source: core.SourceConfig):
        """Load the records of a gek-online source as dicts."""

        res = gws.lib.net.http_request(source.url, params=dict(source.params or {}), verify=False)
        res.raise_if_failed()
        xml = gws.lib.xmlx.from_string((res.text or '').strip(), gws.XmlOptions(removeNamespaces=True))

        rs = []

        for node in xml.findall('Vorgang'):
            rec = {}
            for tag in node:
                rec[tag.name] = tag.text
            rs.append(rec)

        return rs

    def _transform(self, recs, instance_name):
        """Add uids and point geometries to records, skipping duplicate uids."""

        recs2 = []
        points = set()
        uids = set()

        for rec in recs:
            if 'X' not in rec or 'Y' not in rec:
                continue

            xy = self._free_point(
                float(rec.pop('X')),
                float(rec.pop('Y')),
                points,
            )
            points.add(xy)

            rec['instance'] = instance_name

            uid = instance_name + '_' + str(rec['ObjectID'])
            if uid in uids:
                gws.log.warning(f'non-unique {uid=} ignored')
                continue
            uids.add(uid)
            rec['uid'] = uid

            shape = gws.lib.shape.from_geojson(
                {'type': 'Point', 'coordinates': xy},
                self.crs,
            )
            rec['wkb_geometry'] = shape.to_ewkb_hex()

            recs2.append(rec)

        return recs2

    def _free_point(self, x, y, points):
        """Apply the position correction to a point."""

        if not self.position:
            return x, y

        # move a point by specified offsets

        x = round(x, 3) + self.position.offsetX
        y = round(y, 3) + self.position.offsetY

        if (x, y) not in points:
            return x, y

        # if two or more points share the same XY,
        # arrange them in a circle around XY

        distance = self.position.distance
        angle = self.position.angle

        if not distance:
            return x, y

        for a in range(0, 360, angle):
            a = math.radians(a)
            xa = round(x + distance * math.cos(a))
            ya = round(y + distance * math.sin(a))

            if (xa, ya) not in points:
                return xa, ya

        return x, y

    def _write(self, recs):
        """Recreate the index table and insert the records."""
        columns = [
            sa.Column('uid', sa.Text, primary_key=True),
            sa.Column('ObjectID', sa.Text),
            sa.Column('AntragsartBez', sa.Text),
            sa.Column('AntragsartID', sa.Integer, index=True),
            sa.Column('Darstellung', sa.Text),
            sa.Column('Massnahme', sa.Text),
            sa.Column('SystemNr', sa.Text),
            sa.Column('status', sa.Text),
            sa.Column('Tooltip', sa.Text),
            sa.Column('UrlFV', sa.Text),
            sa.Column('UrlOL', sa.Text),
            sa.Column('Verfahren', sa.Text),
            sa.Column('instance', sa.Text),
            sa.Column('wkb_geometry', sa.geo.Geometry(geometry_type='POINT', srid=self.crs.srid), index=True),
        ]

        schema, name = self.db.split_table_name(self.tableName)
        sa_meta = sa.MetaData(schema=schema)
        table = sa.Table(name, sa_meta, *columns, schema=schema)

        with self.db.connect() as conn:
            table.drop(conn.saConn, checkfirst=True)
            table.create(conn.saConn)
            conn.commit()
            if recs:
                conn.execute(sa.insert(table), recs)
                conn.commit()

        gws.log.info(f'saved {len(recs)} records in {schema}.{name}')
