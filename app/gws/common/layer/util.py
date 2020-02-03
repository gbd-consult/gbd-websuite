import gws
import gws.types as t


def add_layers_to_object(target: t.IObject, layer_configs):
    ls = []
    for p in layer_configs:
        try:
            ls.append(target.add_child('gws.ext.layer', p))
        except Exception as e:
            uid = gws.get(p, 'uid')
            gws.log.error(f'FAILED LAYER: parent={target.uid!r} layer={uid!r} error={e!r}')
            gws.log.exception()
            # @TODO: should be an option
            raise
    return ls
