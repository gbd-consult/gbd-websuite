"""Models and features."""

import gws
import gws.base.feature

from . import auth


def context(**kwargs) -> gws.ModelContext:
    kwargs.setdefault('op', gws.ModelOperation.read)
    kwargs.setdefault('user', auth.system_user())
    return gws.ModelContext(**kwargs)


def feature_from_dict(model, atts) -> gws.Feature:
    f = gws.base.feature.new(model=model, record=gws.FeatureRecord(attributes=atts))
    f.attributes = atts
    return f


def feature(model, **atts) -> gws.Feature:
    return feature_from_dict(model, atts)
