import gws.lib.feature

# plain text format

def parse(s, **kwargs):
    return [gws.lib.feature.Feature(attributes={'text': s})]
