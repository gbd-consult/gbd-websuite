import gws
import gws.ows.gml


def parse(s, first_el, **kwargs):
    if 'gml' not in first_el.namespaces and 'gmlx' not in first_el.namespaces:
        return None
    return gws.ows.gml.features_from_xml(s, invert_axis=kwargs.get('invert_axis'))
