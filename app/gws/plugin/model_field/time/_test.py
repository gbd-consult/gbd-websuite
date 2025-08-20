import gws
import gws.config
import gws.base.feature
import gws.test.util as u
import gws.lib.datetimex as dtx


@u.fixture(scope='module')
def model():
    u.pg.create('t', {'id': 'int primary key', 'time': 'TIME'})
    cfg = '''
        models+ { 
            uid "TEST_MODEL" type "postgres" tableName "t"
            fields+ { name "id" type "integer" }
            fields+ { name "time" type "time" } 
        }
    '''
    root = u.gws_root(cfg)
    yield u.cast(gws.Model, root.get('TEST_MODEL'))


##

def test_create(model: gws.Model):
    mc = u.model_context()
    time = dtx.new(2000, 2, 3, 4, 5, 6)

    f = u.feature(model, id=3, time=time)
    model.create_feature(f, mc)
    fs = model.get_features([3], mc)

    assert dtx.to_iso_time_string(fs[0].get('time')) == '04:05:06'


def test_read(model: gws.Model):
    mc = u.model_context()

    ds = [
        dtx.new(2000, 2, 3, 4, 5, 6),
        dtx.new(2001, 12, 13, 14, 15, 16),
    ]
    u.pg.insert('t', [{'id': 1, 'time': ds[0]}, {'id': 2, 'time': ds[1]}])

    fs = model.get_features([1, 2], mc)

    assert dtx.to_iso_time_string(fs[0].get('time')) == '04:05:06'
    assert dtx.to_iso_time_string(fs[1].get('time')) == '14:15:16'


def test_update(model: gws.Model):
    mc = u.model_context()

    ds = [
        dtx.new(2000, 2, 3, 4, 5, 6),
        dtx.new(2001, 12, 13, 14, 15, 16),
    ]
    u.pg.insert('t', [{'id': 1, 'time': ds[0]}, {'id': 2, 'time': ds[1]}])

    new = dtx.new(2009, 9, 19, 11, 22, 33)
    f = u.feature(model, id=1, time=new)
    model.update_feature(f, mc)

    fs = model.get_features([1, 2], mc)
    assert dtx.to_iso_time_string(fs[0].get('time')) == '11:22:33'
    assert dtx.to_iso_time_string(fs[1].get('time')) == '14:15:16'
