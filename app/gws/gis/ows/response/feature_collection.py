import gws
import gws.gis.gml


def parse(s, first_el, **kwargs):
    if 'gml' not in first_el.namespaces and 'gmlx' not in first_el.namespaces:
        return None
    return gws.gis.gml.features_from_xml(s, invert_axis=kwargs.get('invert_axis'))
