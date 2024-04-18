"""Qquery the Gekos-Online server."""

from typing import Optional

import math

import gws
import gws.base.database
import gws.base.shape
import gws.gis.crs
import gws.lib.xmlx
import gws.lib.net
import gws.lib.sa as sa
from gws.lib.console import ProgressIndicator


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


class PositionConfig(gws.Config):
    offsetX: int  #: x-offset for points
    offsetY: int  #: y-offset for points
    distance: int = 0  #: radius for points repelling
    angle: int = 0  #: angle for points repelling


class SourceConfig(gws.Config):
    url: gws.Url  #: gek-online base url
    params: dict  #: additional parameters for gek-online calls
    instance: str  #: instance name for this source


class Config(gws.Config):
    crs: gws.CrsName
    """CRS for gekos data"""
    dbUid: Optional[str]
    """Database provider uid"""
    sources: list[SourceConfig]
    """gek-online instance names"""
    position: Optional[PositionConfig]
    """position correction for points"""
    tableName: str
    """sql table name"""


class Object(gws.Node):
    provider: gws.DatabaseProvider
    tableName: str
    position: PositionConfig
    crs: gws.Crs

    def configure(self):
        self.provider = gws.base.database.provider.get_for(self, ext_type='postgres')
        self.tableName = self.cfg('tableName')
        self.crs = gws.gis.crs.get(self.cfg('crs'))
        self.position = self.cfg('position')

    def create(self):
        recs = self._collect()
        self._write(recs)

    def _collect(self):
        recs = []

        for source in self.cfg('sources'):
            rs = self._load(source)
            gws.log.info(f'loaded {len(rs)} records from {source.instance!r}')
            rs = self._transform(rs, source.instance)
            recs.extend(rs)

        return recs

    def _load(self, source: SourceConfig):
        """Load XML from GekOnline and create record dicts."""

        res = gws.lib.net.http_request(source.url, params=dict(source.params or {}), verify=False)
        res.raise_if_failed()
        xml = gws.lib.xmlx.from_string((res.text or '').strip())

        rs = []

        for node in xml.findall('Vorgang'):
            rec = {}
            for tag in node:
                rec[tag.name] = tag.text
            rs.append(rec)

        return rs

    def _transform(self, recs, instance_name):
        """Compute geometries and uids for record dicts."""

        recs2 = []
        points = set()
        uids = set()

        for rec in recs:
            if 'X' not in rec or 'Y' not in rec:
                continue

            xy = self._free_point(
                float(rec.pop('X')),
                float(rec.pop('Y')),
                points
            )
            points.add(xy)

            rec['instance'] = instance_name

            uid = instance_name + '_' + str(rec['ObjectID'])
            if uid in uids:
                gws.log.warning(f'non-unique {uid=} ignored')
                continue
            uids.add(uid)
            rec['uid'] = uid

            shape = gws.base.shape.from_geojson({
                'type': 'Point',
                'coordinates': xy
            }, self.crs)
            rec['wkb_geometry'] = shape.to_ewkb_hex()

            recs2.append(rec)

        return recs2

    def _free_point(self, x, y, points):
        """Move points around, according to the 'position' config."""

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

        schema, name = self.provider.split_table_name(self.tableName)
        sa_meta = sa.MetaData(schema=schema)
        table = sa.Table(name, sa_meta, *columns, schema=schema)

        with self.provider.engine().connect() as conn:
            table.drop(conn, checkfirst=True)
            table.create(conn)
            conn.commit()
            conn.execute(sa.insert(table).values(recs))
            conn.commit()

        gws.log.info(f'saved {len(recs)} records in {schema}.{name}')
