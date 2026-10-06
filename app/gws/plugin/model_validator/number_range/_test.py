import gws
import gws.test.util as u


@u.fixture(scope='module')
def model():
    cfg = '''
        models+ {
            uid "MODEL_1" type "default"
            fields+ {
                name "field_1" type "integer"
                validators+ {
                    type "numberRange"
                    min { type "static" value 1 }
                    max { type "static" value 10 }
                }
            }
            fields+ {
                name "field_2" type "integer"
                validators+ {
                    type "numberRange"
                    min { type "expression" expression "None" }
                }
            }
            fields+ {
                name "field_3" type "integer"
                validators+ {
                    type "numberRange"
                    max { type "expression" expression "None" }
                }
            }
        }
    '''
    root = u.gws_root(cfg)
    yield u.cast(gws.Model, root.get('MODEL_1'))


def _validate(model, name, val):
    fld = model.field(name)
    f = u.model.feature(model, **{name: val})
    return fld.validators[0].validate(fld, f, u.model.context())


def test_in_range(model: gws.Model):
    assert _validate(model, 'field_1', 1) is True
    assert _validate(model, 'field_1', 10) is True


def test_out_of_range(model: gws.Model):
    assert _validate(model, 'field_1', 0) is False
    assert _validate(model, 'field_1', 11) is False


def test_not_a_number(model: gws.Model):
    assert _validate(model, 'field_1', 'str_1') is False


def test_min_bound_none(model: gws.Model):
    assert _validate(model, 'field_2', 5) is False


def test_max_bound_none(model: gws.Model):
    assert _validate(model, 'field_3', 5) is False
