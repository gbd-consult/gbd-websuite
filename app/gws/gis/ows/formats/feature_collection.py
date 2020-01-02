import gws
import gws.gis.gml


def parse(text, first_el, crs=None, invert_axis=None, **kwargs):
    if 'gml' not in first_el.namespaces and 'gmlx' not in first_el.namespaces:
        return None
    return gws.gis.gml.features_from_xml(text, invert_axis=invert_axis)
