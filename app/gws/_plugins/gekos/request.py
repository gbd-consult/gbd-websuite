"""Qquery the Gekos-Online server."""

import math

import gws
import gws.types as t
import gws.lib.net
import gws.lib.xml2

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


class GekosRequest:
    def __init__(self, options, instance, cache_lifetime=None):
        self.options = options
        self.instance = instance or 'none'
        self.cache_lifetime = cache_lifetime
        self.position = gws.get(self.options, 'position')

    def run(self):
        recs = []
        used_points = set()

        for rec in self.raw_data():

            if 'X' not in rec or 'Y' not in rec:
                continue

            xy = self.free_point(
                float(rec.pop('X')),
                float(rec.pop('Y')),
                used_points)

            used_points.add(xy)

            rec['instance'] = self.instance
            rec['uid'] = self.instance + '_' + str(rec['ObjectID'])
            rec['xy'] = xy

            recs.append(rec)

        return recs

    def raw_data(self):
        src = self.load_data()
        if not src:
            return []

        xml = gws.lib.xml2.from_string(src)
        for node in xml.all('Vorgang'):
            rec = {}
            for tag in node.children:
                rec[tag.name] = tag.text
            yield rec

    def load_data(self):
        params = dict(self.options.params or {})
        res = gws.lib.net.http_request(
            self.options.url,
            params=params
        )
        return (res.text or '').strip()

    def free_point(self, x, y, points):
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
