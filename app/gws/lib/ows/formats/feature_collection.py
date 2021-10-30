import gws
import gws.lib.gml
import gws.types as t


def parse(text, first_el, crs=None, invert_axis=None, **kwargs):
    if 'gml' not in first_el.namespaces and 'gmlx' not in first_el.namespaces:
        return None
    return gws.lib.gml.features_from_xml(text, invert_axis=invert_axis)
