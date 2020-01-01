import gws.tools.misc
import gws.types as t


def shared_provider(klass, obj, cfg):
    uid = cfg.get('url')
    params = cfg.get('params')
    if params:
        uid += '_' + gws.tools.misc.sha256(' '.join(f'{k}={v}' for k, v in sorted(params.items())))
    return obj.create_shared_object(klass, uid, t.Config(uid=uid).extend(cfg))
