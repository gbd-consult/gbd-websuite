import gws
import gws.types as t


def shared_provider(klass, obj, cfg):
    uid = cfg.get('url')
    params = cfg.get('params')
    if params:
        uid += '_' + gws.sha256(' '.join(f'{k}={v}' for k, v in sorted(params.items())))
    return obj.root.create_shared_object(klass, uid, gws.merge(gws.Config(uid=uid), cfg))
