"""Tests for the uom module."""

import gws
import gws.lib.uom as uom
import gws.test.util as u


def test_scale_to_res():
    value = 1000
    assert uom.scale_to_res(value) == value * uom.OGC_M_PER_PX


def test_res_to_scale():
    value = 1000
    assert uom.res_to_scale(value) == int(value / uom.OGC_M_PER_PX)


def test_mm_to_px():
    value = 1000
    ppi = 32
    assert uom.mm_to_px(value, ppi) == (value * ppi) / uom.MM_PER_IN


def test_to_px():
    """
        Test general conversion of UomValue to pixels

        In it's current state we test the ability to convert
        all supported units, not the correctness of these conversions.

        #TODO: Create Tests for correctness of conversions according to
        # whatever standards there are for a given unit.
    """
    # obj_uom = gws.core.types.Uom()
    # TODO use this line to test aginst all units, but all units are not implemented yet
    # units = [attr for attr in dir(obj_uom) if not callable(getattr(obj_uom, attr)) and not attr.startswith("_")]
    # units = [gws.Uom.mm, gws.Uom.px]
    value = 1000
    ppi = 32
    for unit in [name for name in dir(gws.Uom) if not name.startswith('_')]:
        match unit:
            case gws.Uom.mm:
                assert uom.to_px((value, gws.Uom.mm), ppi) == (uom.mm_to_px(value, ppi), gws.Uom.px)
            case gws.Uom.px:
                assert uom.to_px((value, gws.Uom.px), ppi) == (value, gws.Uom.px)
            case _:
                assert True


def test_size_mm_to_px():
    valuex = 123
    valuey = 321
    valuexy = (valuex, valuey)
    ppi = 32
    assert uom.size_mm_to_px(valuexy, ppi) == (uom.mm_to_px(valuex, ppi), uom.mm_to_px(valuey, ppi))


def test_size_to_px():
    valuex = 1
    valuey = 2
    ppi = 123
    for unit in [name for name in dir(gws.Uom) if not name.startswith('_')]:
        match unit:
            case gws.Uom.mm:
                assert uom.size_to_px((valuex, valuey, gws.Uom.mm), ppi) == (
                    uom.mm_to_px(valuex, ppi), uom.mm_to_px(valuey, ppi), gws.Uom.px)
            case gws.Uom.px:
                assert uom.size_to_px((valuex, valuey, gws.Uom.px), ppi) == (valuex, valuey, gws.Uom.px)
            case _:
                assert True


def test_px_to_mm():
    value = 1000
    ppi = 32
    assert uom.px_to_mm(value, ppi) == (value / ppi) * uom.MM_PER_IN


def test_to_mm():
    # TODO similar to test_to_px we could test for all possible units here once their implemented
    value = 1000
    ppi = 32
    for unit in [name for name in dir(gws.Uom) if not name.startswith('_')]:
        match unit:
            case gws.Uom.px:
                assert uom.to_mm((value, gws.Uom.px), ppi) == (uom.px_to_mm(value, ppi), gws.Uom.mm)
            case gws.Uom.mm:
                assert uom.to_mm((value, gws.Uom.mm), ppi) == (value, gws.Uom.mm)
            case _:
                assert True


def test_size_px_to_mm():
    valuex = 1000
    valuey = 2000
    size = valuex, valuey
    ppi = 32
    assert uom.size_px_to_mm(size, ppi) == (uom.px_to_mm(valuex, ppi), uom.px_to_mm(valuey, ppi))


# not sure if this is covered by test_size_mm_to_px()
def test_size_to_px():
    """
        Assuming test_mm_to_px() passes, this should be sufficient
    """
    valuex = 123
    valuey = 321
    valuexy = (valuex, valuey)
    ppi = 32
    assert uom.size_mm_to_px(valuexy, ppi) == (uom.mm_to_px(valuex, ppi), uom.mm_to_px(valuey, ppi))


def test_size_to_mm():
    valuex = 1
    valuey = 2
    ppi = 123
    for unit in [name for name in dir(gws.Uom) if not name.startswith('_')]:
        match unit:
            case gws.Uom.px:
                assert uom.size_to_mm((valuex, valuey, gws.Uom.px), ppi) == (
                    uom.px_to_mm(valuex, ppi), uom.px_to_mm(valuey, ppi), gws.Uom.mm)
            case gws.Uom.mm:
                assert uom.size_to_mm((valuex, valuey, gws.Uom.mm), ppi) == (valuex, valuey, gws.Uom.mm)
            case _:
                assert True



def test_to_str():
    assert uom.to_str((1, gws.Uom.mm)) == '1mm'
    assert uom.to_str((1.0, gws.Uom.mm)) == '1mm'
    assert uom.to_str((1.1, gws.Uom.m)) == '1.1m'


def test_parse():
    inputs = [
        ('1 mm', (1.0, 'mm')),
        ('10 km', (10.0, 'km')),
        ('-10 deg', (-10.0, 'deg')),
        # ('1,2 m',     (1.2, 'm')),
        ('1.245 inch', (1.245, 'in')),
        # ('0.21 us-in',(0.21, 'us-in'))
    ]
    for test, res in inputs:
        assert uom.parse(test) == res


def test_parse_errors():
    with u.raises(ValueError, match=f'missing unit:'):
        uom.parse(1)
    with u.raises(ValueError, match=f'invalid format:'):
        uom.parse('foo 1')
    with u.raises(ValueError, match=f'invalid unit:'):
        uom.parse('1 bar')


