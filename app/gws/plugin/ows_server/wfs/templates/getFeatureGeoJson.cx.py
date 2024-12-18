"""WFS GetFeature template with GeoJSON."""

import gws.base.ows.server as server
import gws.lib.crs
import gws.lib.datetimex as dx
import gws.lib.jsonx
import gws.lib.mime


def main(ta: server.TemplateArgs):
    fc = dict(type='FeatureCollection', features=[])

    for m in ta.featureCollection.members:
        f = dict(type='Feature', properties={})
        for name, val in sorted(m.feature.attributes.items()):
            if name == m.feature.model.uidName:
                f['id'] = val
            elif gws.u.is_atom(val):
                f['properties'][name] = val
            elif dx.is_date(val):
                f['properties'][name] = dx.to_iso_string(val)
        if m.feature.shape():
            f['geometry'] = m.feature.shape().to_geojson()
        fc['features'].append(f)

    return gws.ContentResponse(
        mime=gws.lib.mime.JSON,
        content=gws.lib.jsonx.to_string(fc),
    )
