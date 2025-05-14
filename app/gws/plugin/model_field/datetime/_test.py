import gws
import gws.test.util as u
import gws.lib.datetimex as dtx


@u.fixture(scope='module')
def model():
    u.pg.create('t', {'id': 'int primary key', 'dat': 'timestamptz'})

    cfg = """
        models+ { 
            uid "TEST_MODEL" type "postgres" tableName "t"
            fields+ { name "id" type "integer" }
            fields+ { name "dat" type "date" }
        }
    """

    root = u.gws_root(cfg)
    yield u.cast(gws.Model, root.get('TEST_MODEL'))


def test_create(model: gws.Model):
    mc = u.model_context()

    d = dtx.new(2000, 2, 3, 4, 5, 6)

    f = u.feature(model, id=1, dat=d)
    model.create_feature(f, mc)

    fs = model.get_features([1], mc)
    assert dtx.to_iso_string(fs[0].get('dat')) == dtx.to_iso_string(d)


def test_read(model: gws.Model):
    mc = u.model_context()

    ds = [
        dtx.new(2000, 2, 3, 4, 5, 6),
        dtx.new(2001, 12, 13, 14, 15, 16),
    ]
    u.pg.insert('t', [{'id': 1, 'dat': ds[0]}, {'id': 2, 'dat': ds[1]}])

    fs = model.get_features([1, 2], mc)

    assert dtx.to_iso_string(fs[0].get('dat')) == dtx.to_iso_string(ds[0])
    assert dtx.to_iso_string(fs[1].get('dat')) == dtx.to_iso_string(ds[1])


def test_update(model: gws.Model):
    mc = u.model_context()

    ds = [
        dtx.new(2000, 2, 3, 4, 5, 6),
        dtx.new(2001, 12, 13, 14, 15, 16),
    ]
    u.pg.insert('t', [{'id': 1, 'dat': ds[0]}, {'id': 2, 'dat': ds[1]}])

    new = dtx.new(2009, 9, 19, 11, 22, 33)
    f = u.feature(model, id=1, dat=new)
    model.update_feature(f, mc)

    fs = model.get_features([1, 2], mc)
    assert dtx.to_iso_string(fs[0].get('dat')) == dtx.to_iso_string(new)
