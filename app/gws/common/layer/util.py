import gws
import gws.types as t


def add_layers_to_object(target: t.IObject, layer_configs):
    ls = []
    skip_invalid = target.var('skipInvalidLayers', parent=True)
    for p in _sorted_layer_configs(layer_configs):
        try:
            ls.append(target.create_child('gws.ext.layer', p))
        except Exception as e:
            uid = gws.get(p, 'uid')
            gws.log.error(f'FAILED LAYER: parent={target.uid!r} layer={uid!r} error={e!r}')
            if skip_invalid:
                gws.log.exception()
            else:
                raise
    return ls


def _sorted_layer_configs(layer_configs):
    # sort configs by - 1) positive 'order' ASC 2) no order 3) negative 'order' ASC

    parts = [[], [], []]

    for n, c in enumerate(layer_configs):
        s = c.get('order') or 0
        if s > 0:
            parts[0].append((s, n))
        elif s == 0:
            parts[1].append((s, n))
        elif s < 0:
            parts[2].append((s, n))

    return [
        layer_configs[n]
        for _, n in
        sorted(parts[0]) + parts[1] + sorted(parts[2])
    ]
