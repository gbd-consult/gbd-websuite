import gws
import gws.types as t


def add_layers_to_object(target: t.IObject, layer_configs):
    ls = []
    skip_invalid = target.var('skipInvalidLayers', parent=True)
    for p in layer_configs:
        try:
            ls.append(target.add_child('gws.ext.layer', p))
        except Exception as e:
            uid = gws.get(p, 'uid')
            gws.log.error(f'FAILED LAYER: parent={target.uid!r} layer={uid!r} error={e!r}')
            if skip_invalid:
                gws.log.exception()
            else:
                raise
    return ls
