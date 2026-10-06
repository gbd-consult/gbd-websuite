import gws
import gws.test.util as u


@u.fixture(scope='module')
def model():
    cfg = '''
        models+ {
            uid "MODEL_1" type "default"
            fields+ {
                name "field_1" type "text"
                values+ {
                    type "expression"
                    imports ["os.path"]
                    expression "os.path.join('a', 'b')"
                }
            }
            fields+ {
                name "field_2" type "integer"
                values+ {
                    type "expression"
                    imports ["math"]
                    expression "math.floor(feature.get('field_2') / 2)"
                }
            }
        }
    '''
    root = u.gws_root(cfg)
    yield u.cast(gws.Model, root.get('MODEL_1'))


def _compute(model, name, **atts):
    fld = model.field(name)
    f = u.model.feature(model, **atts)
    return fld.values[0].compute(fld, f, u.model.context())


def test_dotted_import(model: gws.Model):
    assert _compute(model, 'field_1') == 'a/b'


def test_import_and_feature(model: gws.Model):
    assert _compute(model, 'field_2', field_2=7) == 3
